import pathlib
from typing import List, Optional, Sequence, Union
from abc import abstractmethod

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from cmrl.models.util import gaussian_nll
from cmrl.models.layers import ParallelLinear
from cmrl.models.networks.base_network import BaseNetwork


# partial from https://github.com/phlippe/ENCO/blob/main/causal_discovery/multivariable_mlp.py
class ParallelMLP(BaseNetwork):
    _MODEL_FILENAME = "parallel_mlp.pth"

    def __init__(self, network_cfg: DictConfig):
        super().__init__(network_cfg)

    def build_network(self):
        activation_fn_cfg = self.network_cfg.get("activation_fn_cfg", None)
        extra_dims = self.network_cfg.get("extra_dims", None)

        def create_activation():
            if activation_fn_cfg is None:
                return nn.ReLU()
            else:
                return hydra.utils.instantiate(activation_fn_cfg)

        layers = []
        hidden_dims = [self.network_cfg.input_dim] + self.network_cfg.hidden_dims
        for i in range(len(hidden_dims) - 1):
            layers += [ParallelLinear(input_dim=hidden_dims[i], output_dim=hidden_dims[i + 1], extra_dims=extra_dims)]
            layers += [create_activation()]
        layers += [ParallelLinear(input_dim=hidden_dims[-1], output_dim=self.network_cfg.output_dim, extra_dims=extra_dims)]

        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
