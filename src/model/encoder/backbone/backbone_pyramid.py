import os

import torch
import torch.nn.functional as F
from einops import rearrange

from .multiview_transformer import MultiViewFeatureTransformer
from .mvsformer_module.cost_volume import *
from .mvsformer_module.dino.dinov2 import vit_base
from .unimatch.backbone import ResUnet
from .unimatch.position import PositionEmbeddingSine
from .unimatch.utils import merge_splits, split_feature

# TODO: temporary config, should use cfg file
# The global vars
Align_Corners_Range = False

# for align feature
args = {
    "model_type": "DINOv2-base",
    "freeze_vit": True,
    "rescale": 1.0,
    "vit_ch": 768,
    "out_ch": 64,
    "vit_path": "./checkpoints/dinov2_vitb14_pretrain.pth",
    "pretrain_mvspp_path": "",
    "depth_type": ["ce", "ce", "ce", "ce"],
    "fusion_type": "cnn",
    "inverse_depth": True,
    "base_ch": [8, 8, 8, 8],
    "ndepths": [128, 64, 32, 16],
    "feat_chs": [32, 64, 128, 256],
    "depth_interals_ratio": [4.0, 2.67, 1.5, 1.0],
    "decoder_type": "CrossVITDecoder",
    "dino_cfg": {
        "use_flash2_dino": False,
        "softmax_scale": None,
        "train_avg_length": 762,
        "cross_interval_layers": 3,
        "decoder_cfg": {
            "init_values": 1.0,
            "prev_values": 0.5,
            "d_model": 768,
            "nhead": 12,
            "attention_type": "Linear",
            "ffn_type": "ffn",
            "softmax_scale": "entropy_invariance",
            "train_avg_length": 762,
            "self_cross_types": None,
            "post_norm": False,
            "pre_norm_query": True,
            "no_combine_norm": False,
        },
    },
    "FMT_config": {
        "attention_type": "Linear",
        "base_channel": 8 * 4,
        "d_model": 64 * 4,
        "nhead": 4,
        "init_values": 1.0,
        "layer_names": ["self", "cross", "self", "cross"],
        "ffn_type": "ffn",
        "softmax_scale": "entropy_invariance",
        "train_avg_length": 12185,
        "attn_backend": "FLASH2",
        "self_cross_types": None,
        "post_norm": False,
        "pre_norm_query": False,
    },
    "cost_reg_type": ["PureTransformerCostReg", "Normal", "Normal", "Normal"],
    "use_pe3d": True,
    "transformer_config": [
        {
            "base_channel": 8 * 4,
            "mid_channel": 64 * 4,
            "num_heads": 4,
            "down_rate": [2, 4, 4],
            "mlp_ratio": 4.0,
            "layer_num": 6,
            "drop": 0.0,
            "attn_drop": 0.0,
            "position_encoding": True,
            "attention_type": "FLASH2",
            "softmax_scale": "entropy_invariance",
            "train_avg_length": 12185,
            "use_pe_proj": True,
        }
    ],
}


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [split_feature(x, num_splits=attn_splits) for x in features_list]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [merge_splits(x, num_splits=attn_splits) for x in features_splits]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class BackbonePyramid(torch.nn.Module):
    """docstring for BackboneMultiview.
    This function is used to extract the feature of different view
    the CNN is used to extract single view feature
    Transformer is used to extract single&multi view feature
    """

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_self_attn=False,
        no_cross_attn=False,
        num_head=1,
        no_split_still_shift=False,
        no_ffn=False,
        global_attn_fast=True,
        downscale_factor=8,
        use_epipolar_trans=False,
    ):
        super(BackbonePyramid, self).__init__()
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )
        self.backbone = ResUnet()
        self.dino = DinoExtractor()

    def normalize_images(self, images):
        # TODO: should use the normalize for other model
        """Normalize image to match the pretrained GMFlow backbone.
        images: (B, N_Views, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, context):
        # stage 1-> 32x32 stage 2-> 64x64 stage 3-> 128x128 stage 4-> 256x256
        # context = self.gen_all_rs_combination(context)
        imgs = self.normalize_images(context["image"])
        org_imgs = imgs
        B, V, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[3], imgs.shape[4]
        # dinov2 patchsize=14,  0.5 * 14/16
        imgs = imgs.reshape(B * V, 3, H, W)
        dino_feature = self.dino(org_imgs)
        out_feature = self.backbone(imgs, dino_feature)
        trans_feature_in = rearrange(out_feature[0], "(b v) c h w -> b v c h w", b=B).chunk(dim=1, chunks=V)
        trans_feature_in = [f[:, 0] for f in trans_feature_in]
        # add position to features
        trans_feature_in = feature_add_position_list(trans_feature_in, 2, out_feature[0].size(1))
        cur_features_list = self.transformer(trans_feature_in, 2)
        trans_features = torch.stack(cur_features_list, dim=1)  # B V 64 64
        return (out_feature, trans_features)

    def forward(
        self,
        context,
        attn_splits=2,
        return_cnn_features=False,
        epipolar_kwargs=None,
    ):
        """images: (B, N_Views, C, H, W), range [0, 1]"""
        # resolution low to high
        features_list = self.extract_feature(context=context)  # list of features

        return features_list

    def gen_proj_matrices(self, context):
        # extri and intri
        # !!!Attention: Mvsformer++ need w2c so we have to inverse
        inverse = True
        extri = context["extrinsics"].clone().inverse() if inverse else context["extrinsics"].clone()
        intri = context["intrinsics"].clone()

        b, v, _, _ = intri.shape
        _, _, _, h, w = context["image"].shape
        intri = torch.cat(
            [
                torch.cat([intri, torch.zeros((b, v, 1, 3), device=intri.device)], dim=2),
                torch.zeros((b, v, 4, 1), device=intri.device),
            ],
            dim=3,
        )
        intri_org = intri.clone()
        proj_matrices = {"stage{}".format(i): 0 for i in range(1, 5)}
        # four stages and the intrinsics scale by 2x
        for i in range(1, 5):
            intri[..., 0, :] = intri_org[..., 0, :] * w / 2 ** (4 - i)
            intri[..., 1, :] = intri_org[..., 1, :] * h / 2 ** (4 - i)
            proj_matrices["stage{}".format(i)] = torch.stack([extri, intri], dim=2)
        return proj_matrices  # [b, v, 2, 4, 4]

    def gen_all_rs_combination(self, context):
        # according to the input context, output all the reference and source combination of different pictures
        def reverse_and_add_to_bv(input_tensor):
            # input_tensor: [b, v, ...]
            b, v, *_ = input_tensor.shape
            tensor_return_order_list = []
            for i in range(v):
                range_v = list(range(v))
                del range_v[i]
                tensor_return_order_list += [[i] + range_v]
            return_input_tensor = rearrange(
                input_tensor[:, tensor_return_order_list], "b com_n v ... -> (b com_n) v ..."
            )
            return return_input_tensor

        return {k: reverse_and_add_to_bv(v) for k, v in context.items()}


class DinoExtractor(torch.nn.Module):
    def __init__(self, vit_path="./checkpoints/dinov2_vitb14_pretrain.pth"):
        super(DinoExtractor, self).__init__()
        self.vit_cfg = {
            "model_type": "DINOv2-base",
            "freeze_vit": True,
            "rescale": 1.0,
            "vit_ch": 768,
            "out_ch": 64,
            "vit_path": "./checkpoints/dinov2_vitb14_pretrain.pth",
            "pretrain_mvspp_path": "",
            "depth_type": ["ce", "ce", "ce", "ce"],
            "fusion_type": "cnn",
            "inverse_depth": True,
            "base_ch": [8, 8, 8, 8],
            "ndepths": [128, 64, 32, 16],
            "feat_chs": [32, 64, 128, 256],
            "depth_interals_ratio": [4.0, 2.67, 1.5, 1.0],
            "decoder_type": "CrossVITDecoder",
            "dino_cfg": {
                "use_flash2_dino": False,
                "softmax_scale": None,
                "train_avg_length": 762,
                "cross_interval_layers": 3,
                "decoder_cfg": {
                    "init_values": 1.0,
                    "prev_values": 0.5,
                    "d_model": 768,
                    "nhead": 12,
                    "attention_type": "Linear",
                    "ffn_type": "ffn",
                    "softmax_scale": "entropy_invariance",
                    "train_avg_length": 762,
                    "self_cross_types": None,
                    "post_norm": False,
                    "pre_norm_query": True,
                    "no_combine_norm": False,
                },
            },
            "FMT_config": {
                "attention_type": "Linear",
                "base_channel": 8 * 4,
                "d_model": 64 * 4,
                "nhead": 4,
                "init_values": 1.0,
                "layer_names": ["self", "cross", "self", "cross"],
                "ffn_type": "ffn",
                "softmax_scale": "entropy_invariance",
                "train_avg_length": 12185,
                "attn_backend": "FLASH2",
                "self_cross_types": None,
                "post_norm": False,
                "pre_norm_query": False,
            },
            "cost_reg_type": ["PureTransformerCostReg", "Normal", "Normal", "Normal"],
            "use_pe3d": True,
            "transformer_config": [
                {
                    "base_channel": 8 * 4,
                    "mid_channel": 64 * 4,
                    "num_heads": 4,
                    "down_rate": [2, 4, 4],
                    "mlp_ratio": 4.0,
                    "layer_num": 6,
                    "drop": 0.0,
                    "attn_drop": 0.0,
                    "position_encoding": True,
                    "attention_type": "FLASH2",
                    "softmax_scale": "entropy_invariance",
                    "train_avg_length": 12185,
                    "use_pe_proj": True,
                }
            ],
        }
        dino_cfg = args.get("dino_cfg", {})
        self.vit = vit_base(img_size=518, patch_size=14, init_values=1.0, block_chunks=0, ffn_layer="mlp", **dino_cfg)
        self.decoder_vit = CrossVITDecoder(self.vit_cfg)
        if os.path.exists(vit_path):
            state_dict = torch.load(vit_path, map_location="cpu")
            from .mvsformer_module.utils import torch_init_model

            torch_init_model(self.vit, state_dict, key="model")
            print("!!!Successfully load the DINOV2 ckpt from", vit_path)
        else:
            print("!!!No weight in", vit_path, "testing should neglect this.")

    def forward(self, image):
        rescale = 0.4375
        b, v, c, h, w = image.shape
        image = rearrange(image, "b v c h w -> (b v) c h w")
        vit_h, vit_w = int(h * rescale // 14 * 14), int(w * rescale // 14 * 14)
        vit_imgs = F.interpolate(image, (vit_h, vit_w), mode="bicubic", align_corners=False)
        with torch.no_grad():
            vit_out = self.vit.forward_interval_features(vit_imgs)
        vit_out = [vi.reshape(b, v, -1, self.vit.embed_dim) for vi in vit_out]
        vit_shape = [b, v, vit_h // self.vit.patch_size, vit_w // self.vit.patch_size, self.vit.embed_dim]
        vit_feat = self.decoder_vit.forward(vit_out, Fmats=None, vit_shape=vit_shape)
        return vit_feat
