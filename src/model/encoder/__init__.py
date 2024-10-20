from typing import Optional

from .encoder import Encoder
from .encoder_costvolume_pyramid import (
    EncoderCostVolumeCfgPyramid,
    EncoderCostVolumePyramid,
)
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume

ENCODERS = {
    "costvolume_pyramid": (EncoderCostVolumePyramid, EncoderVisualizerCostVolume),
}

EncoderCfg = EncoderCostVolumeCfgPyramid


def get_encoder(cfg: EncoderCfg, decoder) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    encoder.decoder = decoder
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
