import pathlib
from typing import List, Optional, Sequence, Union
from abc import abstractmethod

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from cmrl.models.layers import ParallelLinear
from cmrl.models.networks.base_network import BaseNetwork, create_activation


# partial from https://github.com/phlippe/ENCO/blob/main/causal_discovery/multivariable_mlp.py
class ParallelMLP(BaseNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        extra_dims: Optional[List[int]] = None,
        hidden_dims: Optional[List[int]] = None,
        bias: bool = True,
        init_type: str = "truncated_normal",
        activation_fn_cfg: Optional[DictConfig] = None,
        **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.extra_dims = extra_dims if extra_dims is not None else []
        self.hidden_dims = hidden_dims if hidden_dims is not None else [200, 200, 200, 200]
        self.bias = bias
        self.init_type = init_type
        self.activation_fn_cfg = activation_fn_cfg

        super().__init__(**kwargs)
        self._model_filename = "parallel_mlp.pth"

    def build(self):
        layers = []
        hidden_dims = [self.input_dim] + self.hidden_dims
        for i in range(len(hidden_dims) - 1):
            layers += [
                ParallelLinear(
                    input_dim=hidden_dims[i], output_dim=hidden_dims[i + 1], extra_dims=self.extra_dims, bias=self.bias
                )
            ]
            layers += [create_activation(self.activation_fn_cfg)]
        layers += [
            ParallelLinear(input_dim=hidden_dims[-1], output_dim=self.output_dim, extra_dims=self.extra_dims, bias=self.bias)
        ]

        self._layers = nn.ModuleList(layers)
