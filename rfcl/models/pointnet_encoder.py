"""PointNetEncoder class"""

from dataclasses import dataclass, asdict
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Union, List, Sequence
from .types import NetworkConfig

@dataclass
class STNArchConfig:
    conv_channels: List[int]
    mlp_features: List[int]
    activation: Union[Callable, str] = "relu"

@dataclass
class PointNetEncoderArchConfig:
    features: List[int]
    use_stn: bool = True
    activation: Union[Callable, str] = "relu"
    stn: STNArchConfig = None
    name: str = None

@dataclass
class PointNetEncoderConfig(NetworkConfig):
    type = "pointnet_encoder"
    arch_cfg: PointNetEncoderArchConfig

def default_init(scale: Optional[float] = np.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class STN3d(nn.Module):
    conv_channels: Sequence[int]
    mlp_features: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x):
        # x = jnp.atleast_3d(x) 
        has_batch = len(x.shape) == 3
        for feat in self.conv_channels:
            # x = nn.Conv(feat, (1,), kernel_init=default_init())(x)
            x = nn.Dense(feat, kernel_init=default_init())(x)
            x = self.activation(x)

        x = jnp.max(x, axis=-2, keepdims=False)

        for feat in self.mlp_features:
            x = nn.Dense(feat, kernel_init=default_init())(x)
            x = self.activation(x)

        x = nn.Dense(9, kernel_init=default_init())(x)

        iden = jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.float32)

        x = x + iden
        x = x.reshape((len(x), 3, 3)) if has_batch else x.reshape((3, 3))

        return x


class PointNetEncoder(nn.Module):
    features: Sequence[int]
    use_stn: bool
    activation: Callable[[jnp.ndarray], jnp.ndarray]
    stn: STN3d = None

    # def filter_points(self, x):
    #     # Create a mask where the last element in the last dimension is 0
    #     mask = (x[..., 3] == 0)  # Shape: (batch_size, num_points)
    #     updated_array = jnp.where(mask[..., None], jnp.array([0, 0, 0, 0, 0, 0, 0]), x)
        
    #     return updated_array

    @nn.compact
    def __call__(self, x):
        # x (batch_sz, num_points, 3(or 4, xyzw))
        # x = jnp.atleast_3d(x) 
        # if len(x.shape) == 2:
        #     x = jnp.expand_dims(x, 0) # (batch, num_points, 4)

        # x = self.filter_points(x)
        if self.use_stn:
            R = self.stn(x)
            x = jnp.concatenate(((x[..., :3] @ R.swapaxes(-1, -2)), x[..., 3:]), -1) # (batch_sz, num_points, 4)

        for feat in self.features[:-1]:
            # x = nn.Conv(feat, (1,))(x)
            x = nn.Dense(feat, kernel_init=default_init())(x)
            x = self.activation(x)

        # x = nn.Conv(self.features[-1], (1,))(x)
        x = nn.Dense(self.features[-1], kernel_init=default_init())(x)

        x = jnp.max(x, -2, keepdims=False)

        return x