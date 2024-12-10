"""
Build various base models with configurations
"""
from dataclasses import asdict
from typing import Callable

import flax.linen as nn
from dacite import from_dict

from .mlp import MLP, MLPConfig
from .conv import Conv, ConvConfig
from .pointnet_encoder import PointNetEncoder, PointNetEncoderConfig, STN3d
from .types import NetworkConfig


ACTIVATIONS = dict(relu=nn.relu, gelu=nn.gelu, tanh=nn.tanh, sigmoid=nn.sigmoid, log_softmax=nn.log_softmax)


def activation_to_fn(activation: str) -> Callable:
    if activation is None:
        return None
    if activation in ACTIVATIONS:
        return ACTIVATIONS[activation]
    else:
        raise ValueError(f"{activation} is not handled as an activation. Handled activations are {list(ACTIVATIONS.keys())}")


def build_network_from_cfg(cfg: NetworkConfig):
    if cfg.type == "mlp": # used for feature extractor
        cfg = from_dict(data_class=MLPConfig, data=asdict(cfg))
        cfg.arch_cfg.activation = activation_to_fn(cfg.arch_cfg.activation)
        cfg.arch_cfg.output_activation = activation_to_fn(cfg.arch_cfg.output_activation)
        return MLP(**asdict(cfg.arch_cfg))
    elif cfg.type == "conv": # used for rgb visual encoder
        cfg = from_dict(data_class=ConvConfig, data=asdict(cfg))
        cfg.arch_cfg.activation = activation_to_fn(cfg.arch_cfg.activation)
        return Conv(**asdict(cfg.arch_cfg))
    elif cfg.type == "pointnet_encoder": # used for point cloud visual encoder
        cfg = from_dict(data_class=PointNetEncoderConfig, data=asdict(cfg))
        cfg.arch_cfg.activation = activation_to_fn(cfg.arch_cfg.activation)
        pointnet_cfg_dict = asdict(cfg.arch_cfg)
        if cfg.arch_cfg.use_stn:
            cfg.arch_cfg.stn.activation = activation_to_fn(cfg.arch_cfg.stn.activation)
            pointnet_cfg_dict['stn'] = STN3d(**asdict(cfg.arch_cfg.stn))
            # pointnet_cfg_dict.pop("stn_arch_cfg")
        return PointNetEncoder(**pointnet_cfg_dict)
