from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from cmrl.types import Variable, DiscreteVariable, ContinuousVariable
from cmrl.models.networks.base_network import BaseNetwork, create_activation


class VariableEncoder(BaseNetwork):
    def __init__(
        self,
        variable: Variable,
        node_dim: int,
        activation_fn_cfg: Optional[DictConfig] = None,
    ):
        self.variable = variable
        self.node_dim = node_dim
        self.name = "{}_encoder".format(variable.name)
        self.activation_fn_cfg = activation_fn_cfg

        super(VariableEncoder, self).__init__()
        self._model_filename = "{}.pth".format(self.name)

    def build(self):
        layers = []
        if isinstance(self.variable, ContinuousVariable):
            layers.append(nn.Linear(self.variable.dim, self.node_dim))
        elif isinstance(self.variable, DiscreteVariable):
            layers.append(nn.Linear(self.variable.n, self.node_dim))
        else:
            raise NotImplementedError

        layers.append(create_activation(self.activation_fn_cfg))
        self._layers = nn.ModuleList(layers)


class VariableDecoder(BaseNetwork):
    def __init__(
        self,
        variable: Variable,
        node_dim: int,
        normal_distribution: bool = False,
        activation_fn_cfg: Optional[DictConfig] = None,
    ):
        self.variable = variable
        self.node_dim = node_dim
        self.normal_distribution = normal_distribution
        self.name = "{}_decoder".format(variable.name)
        self.activation_fn_cfg = activation_fn_cfg

        super(VariableDecoder, self).__init__()
        self._model_filename = "{}.pth".format(self.name)

    def build(self):
        layers = []
        if isinstance(self.variable, ContinuousVariable):
            if self.normal_distribution:
                layers.append(nn.Linear(self.node_dim, self.variable.dim * 2))
            else:
                layers.append(nn.Linear(self.node_dim, self.variable.dim))
        elif isinstance(self.variable, DiscreteVariable):
            layers.append(nn.Linear(self.node_dim, self.variable.n))
            layers.append(nn.Softmax())
        else:
            raise NotImplementedError

        self._layers = nn.ModuleList(layers)
