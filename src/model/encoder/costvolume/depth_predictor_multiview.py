import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ...types import Gaussians
from ..backbone.unimatch.backbone import ResidualBlock
from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(b, h, w, homogeneous=True, device=depth.device)  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(1, 1, d, 1) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(b, 3, d, h * w)  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(b, c, d, h, w)  # [B, C, D, H, W]

    return warped_feature


def prepare_feat_proj_data_lists(features, intrinsics, extrinsics, near, far, num_samples):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().detach().inverse() @ extrinsics[:, v0].clone().detach()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)

    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0].clone().detach()
        pose_tgt = extrinsics[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref
        # pose_list[1->0, 0->1]
        try:
            pose_curr_lists = [
                torch.cat((pose, pose.inverse()), dim=0),
            ]
        except:
            pose = pose.float()
            pose_curr_lists = [
                torch.cat((pose, pose.inverse()), dim=0),
            ]
    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device) * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr


class DepthPredictorMultiViewPyramid(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        **kwargs,
    ):
        super(DepthPredictorMultiViewPyramid, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # attention 0, 1, 2 -> 64*64, 128*128, 256*256
        channel_list = [32, 64, 128]
        depth_candi_list = [128, 64, 32]
        self.depth_candi_list = depth_candi_list
        depth_predictor_list = []
        for stage_id in range(len(channel_list)):
            depth_predictor_list.append(
                DepthPredictorRefine(
                    channel_list,
                    depth_candi_list,
                    depth_unet_feat_dim,
                    depth_unet_attn_res,
                    depth_unet_channel_mult,
                    gaussian_raw_channels,
                    gaussians_per_pixel,
                    num_views,
                    stage_id,
                )
            )
        self.depth_predictor = nn.ModuleList(depth_predictor_list)
        self.modulater = Modulater(2 * 83, 32, mod="mul")

    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        encoder=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""

        # format the input
        in_feats = features
        cnn_feature, trans_feature = features[0], features[1]
        b, v, c, h, w = trans_feature.shape
        cnn_feature = [rearrange(f, "(b v) c h w -> (v b) c h w", b=b) for f in cnn_feature]
        # for different resolution's depth and gaussians
        # attention: it returns b v
        depths, pre_depth = None, None
        result_dict = {"stage0": {}, "stage1": {}, "stage2": {}}
        gaussian_dict = {k: {} for k in result_dict.keys()}
        for i in range(len(self.depth_predictor)):
            init = i == 0
            depth_size = cnn_feature[i].size()[-2:]
            reverse_interval = 1 / self.depth_candi_list[0]

            disp_candi_curr = self.generate_disp_candi_curr(
                near, far, self.depth_candi_list[i], depth_size, reverse_interval, pre_depth=pre_depth, init=init
            )

            depth_predictor = self.depth_predictor[i]
            if i != 0:
                trans_feature_i = F.interpolate(
                    trans_feature.view(b * v, c, h, w), size=depth_size, mode="bilinear", align_corners=True
                ).view(b, v, c, depth_size[0], depth_size[1])
                # render and get the pre stage rendered images
                pre_stage_gaussian = gaussian_dict[f"stage{i-1}"]["gaussians"]
                pre_stage_context_image = encoder.decoder.forward(
                    pre_stage_gaussian,
                    extrinsics,
                    intrinsics,
                    near,
                    far,
                    extra_info["images"].shape[-2:],
                    depth_mode=None,
                )
                pre_stage_context_image = pre_stage_context_image.color  # b v c h w
                result_dict[f"stage{i-1}"]["render_image"] = pre_stage_context_image
                pre_stage_context_image = rearrange(pre_stage_context_image, "b v c h w -> (v b) c h w")
                pre_stage_residual = (extra_info["images"] - pre_stage_context_image).abs()
            else:
                trans_feature_i = trans_feature
                pre_stage_residual = None

            depths, densities, raw_gaussians, coarse_disps, pdf_max = depth_predictor(
                trans_feature_i,
                cnn_feature[i],
                intrinsics,
                extrinsics,
                near,
                far,
                disp_candi_curr,
                extra_info,
                pre_stage_residual,
            )
            if depths is not None:
                pre_depth = 1 / (
                    rearrange(depths, "b v (h w) () () -> (v b) () h w", h=depth_size[0], w=depth_size[1]) + 1e-8
                )
            extra_info["pdf_max"] = pdf_max
            extra_info["coarse_disps"] = rearrange(depths, "b v (h w) () () -> (v b) () h w", h=depth_size[0])
            result_dict[f"stage{i}"]["depths"] = depths
            result_dict[f"stage{i}"]["densities"] = densities
            result_dict[f"stage{i}"]["raw_gaussians"] = raw_gaussians
            result_dict[f"stage{i}"]["coarse_disps"] = coarse_disps
            image_size = cnn_feature[i].shape[-2:]
            return_gaussians, scales, rotations = encoder.convert_to_gaussians_single_stge(
                result_dict[f"stage{i}"]["raw_gaussians"],
                result_dict[f"stage{i}"]["densities"],
                result_dict[f"stage{i}"]["depths"],
                image_size,
                extrinsics,
                intrinsics,
                extra_info["global_step"],
                opacity_multiplier=1.0,
                stage_id=i,
            )
            if i != 0:
                mask = rearrange(pre_stage_residual.mean(dim=1, keepdim=True), "(v b) c h w -> (b v) c h w", b=b)
                raw_gaussians_pre, raw_gaussians_now, mask_return = self.extract_raw_guassians(
                    raw_gaussians, i, result_dict, mask, gaussian_dict, densities, with_density=True
                )
                densi_weight = self.modulater(raw_gaussians_pre, raw_gaussians_now, mask_return, v=v)
                concat_gaussians.opacities = concat_gaussians.opacities * densi_weight[..., 0]
            concat_gaussians = return_gaussians if i == 0 else self.concat_gaussians(concat_gaussians, return_gaussians)
            concat_scales = scales if i == 0 else torch.cat([concat_scales, scales], dim=1)
            concat_rotations = rotations if i == 0 else torch.cat([concat_rotations, rotations], dim=1)
            gaussian_dict[f"stage{i}"]["gaussians"] = concat_gaussians
            gaussian_dict[f"stage{i}"]["depths"] = depths
            gaussian_dict[f"stage{i}"]["scales"] = concat_scales
            gaussian_dict[f"stage{i}"]["rotations"] = concat_rotations
        return gaussian_dict, result_dict

    def concat_gaussians(self, ga: Gaussians, gb: Gaussians):
        return Gaussians(
            torch.cat([ga.means, gb.means], dim=1),
            torch.cat([ga.covariances, gb.covariances], dim=1),
            torch.cat([ga.harmonics, gb.harmonics], dim=1),
            torch.cat([ga.opacities, gb.opacities], dim=1),
        )

    def generate_disp_candi_curr(
        self,
        near,
        far,
        num_samples,
        depth_size,
        reverse_interval: float = 0.0,
        shift_range: float = 2.7,
        pre_depth=None,
        init=False,
    ):
        # return the candi curr with [vb c h w]
        assert (not init and pre_depth is not None) or (init and pre_depth is None)
        h, w = depth_size[0], depth_size[1]
        if init:
            # prepare depth bound (inverse depth) [v*b, d]
            min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
            max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
            depth_candi_curr = min_depth + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device) * (
                max_depth - min_depth
            )
            depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d h w", h=h, w=w)  # [vxb, d, 1, 1]
        else:
            assert pre_depth is not None
            device = pre_depth.device
            vb = pre_depth.size(0)
            shift_range_reverse = reverse_interval * shift_range
            pre_depth = F.interpolate(pre_depth, size=depth_size, mode="bilinear", align_corners=True)
            depth_candi_range = torch.linspace(-shift_range_reverse, shift_range_reverse, num_samples, device=device)
            depth_candi_range = repeat(depth_candi_range, "d -> vb d h w", vb=vb, h=h, w=w)
            depth_candi_curr = pre_depth + depth_candi_range
        return depth_candi_curr

    def extract_raw_guassians(
        self, raw_gaussians, stage_id, result_dict, mask, gaussian_dict, densities_now, with_density=False
    ):
        b, v, r, c = raw_gaussians.shape
        h, w = int(r**0.5), int(r**0.5)
        raw_gaussians_pre, raw_gaussians_now, mask_list = [], [], []
        for j in range(stage_id):
            raw_gaussians_pre_j = result_dict[f"stage{j}"]["raw_gaussians"]
            raw_gaussians_pre.append(rearrange(raw_gaussians_pre_j, "b v r c -> b (v r) c"))
            r_pre = raw_gaussians_pre_j.shape[2]
            h_pre, w_pre = int(r_pre**0.5), int(r_pre**0.5)

            raw_gaussians_now_j = F.interpolate(
                rearrange(raw_gaussians, "b v (h w) c -> (b v) c h w", h=h),
                size=(h_pre, w_pre),
                mode="bilinear",
                align_corners=True,
            )
            if with_density:
                densities_now_reshape = F.interpolate(
                    rearrange(densities_now, "b v (h w) c1 c2-> (b v) (c1 c2) h w", h=h),
                    size=(h_pre, w_pre),
                    mode="bilinear",
                    align_corners=True,
                )
                raw_gaussians_now_j = torch.cat([raw_gaussians_now_j[:, 2:], densities_now_reshape], dim=1)
            raw_gaussians_now.append(rearrange(raw_gaussians_now_j, "(b v) c h w -> b (v h w) c", b=b))
            mask_list.append(
                rearrange(
                    F.interpolate(mask, size=(h_pre, w_pre), mode="bilinear", align_corners=True),
                    "(b v) c h w -> b (v h w) c",
                    b=b,
                )
            )
        raw_gaussians_pre = torch.cat(raw_gaussians_pre, dim=1)
        if with_density:
            raw_gaussians_pre = torch.cat(
                [
                    raw_gaussians_pre[..., 2:],
                    gaussian_dict[f"stage{int(stage_id-1)}"]["gaussians"].opacities[..., None],
                ],
                dim=-1,
            )
        raw_gaussians_now = torch.cat(raw_gaussians_now, dim=1)
        mask_return = torch.cat(mask_list, dim=1)
        return raw_gaussians_pre, raw_gaussians_now, mask_return


class CorrRefineNet(nn.Module):
    def __init__(
        self,
        input_channels,
        channels,
        num_depth_candidates,
        num_views,
        no_self_attn=False,
        use_cross_view_self_attn=True,
    ):
        super(CorrRefineNet, self).__init__()
        modules = [
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=(4,),
                channel_mult=(1, 1, 1),
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=use_cross_view_self_attn,
                no_self_attn=no_self_attn,
            ),
            nn.Conv2d(channels, num_depth_candidates, 3, 1, 1),
        ]
        self.corr_refine_net = nn.Sequential(*modules)
        # cost volume u-net skip connection
        self.regressor_residual = nn.Conv2d(input_channels, num_depth_candidates, 1, 1, 0)

    def forward(self, raw_correlation_in):
        raw_correlation = self.corr_refine_net(raw_correlation_in)  # (vb d h w)
        # apply skip connection
        raw_correlation = raw_correlation + self.regressor_residual(raw_correlation_in)
        return raw_correlation


class DepthPredictorRefine(nn.Module):
    # Include the prediction and refinement of depth
    # Attention 0, 1, 2 -> 64*64, 128*128, 256*256
    # Channel_list = [32, 64, 128]
    # Depth_candi_list = [128, 64, 32]
    def __init__(
        self,
        channel_list,
        depth_candi_list,
        depth_unet_feat_dim,
        depth_unet_attn_res,
        depth_unet_channel_mult,
        gaussian_raw_channels,
        gaussians_per_pixel,
        num_views: int,
        stage_id: int,
    ):
        super(DepthPredictorRefine, self).__init__()
        self.channel_list = channel_list
        self.depth_candi_list = depth_candi_list
        self.num_depth_candidates = depth_candi_list[stage_id]
        self.num_views = num_views
        self.gaussians_per_pixel = gaussians_per_pixel
        stage_num = len(self.channel_list)
        channel_stage_id = int(stage_num - 1 - stage_id)
        self.stage_id = stage_id
        self.channel_stage_id = channel_stage_id
        # only the first stage use the warp and corr refine
        depth_unet_feat_dim = depth_unet_feat_dim * 2 ** (2 - stage_id)
        if self.stage_id == 0:
            self.corr_refine_net = CorrRefineNet(
                input_channels=depth_candi_list[stage_id] + channel_list[channel_stage_id],
                num_views=num_views,
                channels=channel_list[-1],
                num_depth_candidates=depth_candi_list[stage_id],
                no_self_attn=self.stage_id != 0,
                use_cross_view_self_attn=self.stage_id == 0,
            )
            # lowres depth predictor
            self.raw_depth_head = nn.Sequential(
                nn.Conv2d(depth_candi_list[stage_id], depth_candi_list[stage_id] * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(depth_candi_list[stage_id] * 2, depth_candi_list[stage_id], 3, 1, 1),
            )
        self.upsampler = nn.Sequential(
            nn.Conv2d(channel_list[channel_stage_id] + channel_list[-1], channel_list[channel_stage_id], 3, 1, 1),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(channel_list[channel_stage_id], depth_unet_feat_dim, 3, 1, 1)
        # Depth refinement: 2D U-Net
        channels = depth_unet_feat_dim
        self.refine_unet = nn.Sequential(
            nn.Conv2d(
                5 + depth_unet_feat_dim + channel_list[channel_stage_id]
                if self.stage_id == 0
                else 8 + depth_unet_feat_dim + channel_list[channel_stage_id],
                channels,
                3,
                1,
                1,
            ),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=depth_unet_attn_res,
                channel_mult=depth_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
        )
        # Gaussians prediction: covariance, color
        # gau_in = depth_unet_feat_dim + 3 + channel_list[channel_stage_id] if self.stage_id == 0 \
        #     else depth_unet_feat_dim + 3 + channel_list[channel_stage_id] + 3

        gau_in = (
            depth_unet_feat_dim + 3 + channel_list[channel_stage_id]
            if self.stage_id == 0
            else depth_unet_feat_dim + 3 + channel_list[channel_stage_id] + 3
        )
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1),
        )
        # Gaussians prediction: centers, opacity
        channels = depth_unet_feat_dim
        disps_models = [
            nn.Conv2d(channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
        ]
        self.to_disparity = nn.Sequential(*disps_models)
        # The spacial attention module
        if stage_id == 2:
            self.lim_surf = LimSurfacePredictor(channels, channels * 2, lim_dis=0.1)

    def forward(
        self,
        trans_feature,
        cnn_feature,
        intrinsics,
        extrinsics,
        near,
        far,
        disp_candi_curr,
        extra_info,
        pre_stage_residual,
    ):
        # only warp the lowest resolution trans_feature and for the lowest resolution gaussian
        b, v, c, h, w = trans_feature.shape

        if self.stage_id == 0:
            feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = prepare_feat_proj_data_lists(
                trans_feature,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
            # cost volume constructions
            feat01 = feat_comb_lists[0]
            raw_correlation_in_lists = []
            for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                # sample feat01 from feat10 via camera projection pose[0->1, 1->0]
                feat01_warped = warp_with_pose_depth_candidates(
                    feat10,
                    intr_curr,
                    pose_curr,
                    1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                    warp_padding_mode="zeros",
                )  # [B, C, D, H, W]
                # calculate similarity
                raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(1) / (c**0.5)  # [vB, D, H, W]
                raw_correlation_in_lists.append(raw_correlation_in)
            # average all cost volumes
            raw_correlation_in = torch.mean(
                torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
            )  # [vxb d, h, w]
            # refine cost volume via 2D u-net
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
            raw_correlation = self.corr_refine_net(raw_correlation_in)
            # softmax to get coarse depth and density
            pdf = F.softmax(self.raw_depth_head(raw_correlation), dim=1)  # [2xB, D, H, W]
            coarse_disps = (
                (disp_candi_curr * pdf)
                .sum(dim=1, keepdim=True)
                .clamp(
                    1.0 / rearrange(far, "b v -> (v b) () () ()"),
                    1.0 / rearrange(near, "b v -> (v b) () () ()"),
                )
            )  # (vb, 1, h, w)
            pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
        else:
            pdf_max = F.interpolate(
                extra_info["pdf_max"], size=cnn_feature.shape[-2:], mode="bilinear", align_corners=True
            )
            coarse_disps = 1 / F.interpolate(
                extra_info["coarse_disps"], size=cnn_feature.shape[-2:], mode="bilinear", align_corners=True
            )
            feat01 = rearrange(trans_feature, "b v c h w -> (v b) c h w")
            pre_stage_residual = F.interpolate(
                pre_stage_residual, scale_factor=0.5**self.channel_stage_id, mode="bilinear", align_corners=True
            )
        # depth refinement
        proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_feature), dim=1))
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        extra_img = F.interpolate(
            extra_info["images"], scale_factor=0.5**self.channel_stage_id, mode="bilinear", align_corners=True
        )
        if self.stage_id == 0:
            refine_out = self.refine_unet(
                torch.cat([extra_img, proj_feature, coarse_disps, pdf_max, proj_feat_in_fullres], dim=1)
            )
        else:
            refine_out = self.refine_unet(
                torch.cat(
                    [extra_img, proj_feature, coarse_disps, pdf_max, proj_feat_in_fullres, pre_stage_residual], dim=1
                )
            )
        # gaussians head
        if self.stage_id == 0:
            raw_gaussians_in = [refine_out, extra_img, proj_feat_in_fullres]
        else:
            raw_gaussians_in = [refine_out, extra_img, proj_feat_in_fullres, pre_stage_residual]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b)
        # delta fine depth and density
        delta_disps_density = self.to_disparity(refine_out)

        delta_disps, raw_densities = delta_disps_density.split(self.gaussians_per_pixel, dim=1)

        # combine coarse and fine info and match shape
        densities = repeat(
            F.sigmoid(raw_densities),
            "(v b) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        )
        if self.stage_id != 0:
            fine_disps = (coarse_disps + delta_disps).clamp(
                1.0 / rearrange(far, "b v -> (v b) () () ()"),
                1.0 / rearrange(near, "b v -> (v b) () () ()"),
            )
            if self.stage_id == 2:
                fine_disps = self.lim_surf(
                    refine_out,
                    coarse_disps,
                    rearrange(near, "b v -> (v b) () () ()"),
                    rearrange(far, "b v -> (v b) () () ()"),
                )
        else:
            fine_disps = coarse_disps.clamp(
                1.0 / rearrange(far, "b v -> (v b) () () ()"),
                1.0 / rearrange(near, "b v -> (v b) () () ()"),
            )
        depths = 1.0 / fine_disps
        depths = repeat(
            depths,
            "(v b) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        )

        return depths, densities, raw_gaussians, coarse_disps, pdf_max


class Modulater(nn.Module):
    def __init__(self, in_channel, mid_channel, mod="mul"):
        super(Modulater, self).__init__()
        self.mod = mod
        if self.mod == "mul":
            self.reduce_dim_mlp = nn.Sequential(
                nn.Linear(in_channel, mid_channel),
                nn.InstanceNorm2d(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, mid_channel),
                nn.InstanceNorm2d(mid_channel),
                nn.GELU(),
            )
            self.gen_weight = nn.Sequential(
                nn.InstanceNorm2d(mid_channel),
                nn.Linear(mid_channel, mid_channel),
                nn.InstanceNorm2d(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, 1),
                nn.Sigmoid(),
            )
        elif self.mod == "cat":
            self.extract_mask_feature = nn.Sequential(
                ResidualBlock(3, mid_channel), ResidualBlock(mid_channel, mid_channel)
            )
            self.gen_weight = nn.Sequential(
                nn.InstanceNorm2d(mid_channel + in_channel),
                nn.Linear(mid_channel + in_channel, mid_channel),
                nn.InstanceNorm2d(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, mid_channel),
                nn.InstanceNorm2d(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, mid_channel),
                nn.InstanceNorm2d(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, 1),
                nn.Sigmoid(),
            )

    def forward(self, raw_gaussians_pre, raw_gaussians_now, mask, v=2):
        if self.mod == "mul":
            raw_gaussians_cat = self.reduce_dim_mlp(torch.cat([raw_gaussians_pre, raw_gaussians_now], dim=-1))
            raw_gaussians_mid = raw_gaussians_cat * mask
            weight = self.gen_weight(raw_gaussians_mid)
        elif self.mod == "cat":
            raw_gaussians_cat = torch.cat([raw_gaussians_pre, raw_gaussians_now, mask], dim=-1)
            weight = self.gen_weight(raw_gaussians_cat)
        else:
            raise NotImplementedError
        return weight


class LimSurfacePredictor(nn.Module):
    def __init__(self, in_channel, mid_channel, lim_dis=0.5):
        super(LimSurfacePredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, 1, 1)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channel + 2, mid_channel, 1, 1, 0)
        self.gelu2 = nn.GELU()
        self.conv3 = nn.Conv2d(mid_channel, 1, 1, 1, 0)
        self.lim_dis = lim_dis

    # Attention! Depth is in the reverse space!
    def forward(self, feature, depth, near, far):
        depth_min = 1 / (1 / depth * (1 - self.lim_dis)).clamp(min=near, max=far)
        depth_max = 1 / (1 / depth * (1 + self.lim_dis)).clamp(min=near, max=far)
        feature_mid = self.gelu1(self.conv1(feature))
        feature_mix = torch.cat([feature_mid, depth_min, depth_max], dim=1)
        depth_shift = F.sigmoid(self.conv3(self.gelu2(self.conv2(feature_mix))))
        return_depth = depth_max + depth_shift * (depth_min - depth_max)
        return return_depth
