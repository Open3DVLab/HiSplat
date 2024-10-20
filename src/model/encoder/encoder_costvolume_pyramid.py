from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ...global_cfg import get_cfg
from ..types import Gaussians
from .backbone import BackbonePyramid
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .costvolume.depth_predictor_multiview import DepthPredictorMultiViewPyramid
from .encoder import Encoder
from .visualization.encoder_visualizer_costvolume_cfg import (
    EncoderVisualizerCostVolumeCfg,
)


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfgPyramid:
    name: str
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]


class EncoderCostVolumePyramid(Encoder):
    backbone: BackbonePyramid
    depth_predictor: DepthPredictorMultiViewPyramid
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        self.backbone = BackbonePyramid(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
        )

        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == "train":
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = {}
                for k, v in unimatch_pretrained_model.items():
                    if k in self.backbone.state_dict():
                        updated_state_dict[k] = v
                    else:
                        possible_k = "backbone.encoder." + ".".join(k.split(".")[1:])
                        if possible_k in self.backbone.state_dict():
                            updated_state_dict["backbone.encoder." + possible_k] = v
                updated_state_dict = OrderedDict(updated_state_dict)
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                self.backbone.load_state_dict(updated_state_dict, strict=False)
        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiViewPyramid(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity. default is pdf
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        # Encode the context images.
        epipolar_kwargs = None
        features_list = self.backbone(
            context,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        # Sample depths from the resulting features.
        in_feats = features_list
        extra_info = {}
        extra_info["images"] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        extra_info["global_step"] = global_step
        gpp = self.cfg.gaussians_per_pixel
        gaussian_dict, result_dict = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            encoder=self,
        )
        return gaussian_dict, result_dict

    def convert_to_gaussians(self, result_dict, context, features_list, global_step, visualization_dump):
        stage_num = len(result_dict)
        gaussian_dict = {k: {} for k in result_dict.keys()}
        device = context["image"].device
        for i in range(stage_num):
            raw_gaussians = result_dict[f"stage{i}"]["raw_gaussians"]
            densities = result_dict[f"stage{i}"]["densities"]
            depths = result_dict[f"stage{i}"]["depths"]
            h, w = features_list[0][i].shape[-2:]
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            offset_xy = gaussians[..., :2].sigmoid()  # [offset: 2, scales: 3, rotation: 4, sh: 3*25 ]
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size  # maximum change 0.5 pixel, normed xy ray
            gpp = self.cfg.gaussians_per_pixel
            gaussians, scales = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),  # 1 2 4096 1 2
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
            )

            # Dump visualizations if needed.
            if visualization_dump is not None:
                visualization_dump["depth"] = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)
                visualization_dump["scales"] = rearrange(gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
                visualization_dump["rotations"] = rearrange(
                    gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                )

            # Optionally apply a per-pixel opacity.
            opacity_multiplier = 1
            scales = rearrange(scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
            rotations = rearrange(gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw")
            gaussian_dict[f"stage{i}"]["gaussians"] = Gaussians(
                rearrange(
                    gaussians.means.float(),
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances.float(),
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics.float(),
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    (opacity_multiplier * gaussians.opacities).float(),
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
            gaussian_dict[f"stage{i}"]["depths"] = depths
            gaussian_dict[f"stage{i}"]["scales"] = scales
            gaussian_dict[f"stage{i}"]["rotations"] = rotations
        return gaussian_dict

    def convert_to_gaussians_single_stge(
        self,
        raw_gaussians,
        densities,
        depths,
        image_size,
        extrinsics,
        intrinsics,
        global_step,
        opacity_multiplier=1.0,
        stage_id=0,
    ):
        device = raw_gaussians.device
        h, w = image_size[0], image_size[1]
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()  # [offset: 2, scales: 3, rotation: 4, sh: 3*25 ]
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size  # maximum change 0.5 pixel, normed xy ray
        gpp = self.cfg.gaussians_per_pixel
        gaussians, scales = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),  # 1 2 4096 1 2
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
            stage_id=stage_id,
        )

        # Optionally apply a per-pixel opacity.
        scales = rearrange(scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
        rotations = rearrange(gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw")
        return_gaussians = Gaussians(
            rearrange(
                gaussians.means.float(),
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances.float(),
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics.float(),
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                (opacity_multiplier * gaussians.opacities).float(),
                "b v r srf spp -> b (v r srf spp)",
            ),
        )
        return return_gaussians, scales, rotations

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
