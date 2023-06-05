from typing import List, Optional

import torch.nn as nn
from omegaconf import DictConfig

from cmrl.utils.variables import Variable, DiscreteVariable, ContinuousVariable, BinaryVariable, RadianVariable
from cmrl.models.networks.base_network import BaseNetwork, create_activation
from cmrl.models.layers import RadianLayer


class VariableEncoder(BaseNetwork):
    def __init__(
        self,
        variable: Variable,
        output_dim: int = 100,
        hidden_dims: Optional[List[int]] = None,
        bias: bool = True,
        activation_fn_cfg: Optional[DictConfig] = None,
    ):
        self.variable = variable
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.bias = bias
        self.activation_fn_cfg = activation_fn_cfg

        self.name = "{}_encoder".format(variable.name)

        super(VariableEncoder, self).__init__()
        self._model_filename = "{}.pth".format(self.name)

    def build(self):
        layers = []
        if len(self.hidden_dims) == 0:
            hidden_dim = self.output_dim
        else:
            hidden_dim = self.hidden_dims[0]

        if isinstance(self.variable, ContinuousVariable):
            layers.append(nn.Linear(self.variable.dim, hidden_dim))
        elif isinstance(self.variable, RadianVariable):
            layers.append(RadianLayer())
            layers.append(nn.Linear(self.variable.dim, hidden_dim))
        elif isinstance(self.variable, DiscreteVariable):
            layers.append(nn.Linear(self.variable.n, hidden_dim))
        elif isinstance(self.variable, BinaryVariable):
            layers.append(nn.Linear(1, hidden_dim))
        else:
            raise NotImplementedError("Type {} is not supported by VariableEncoder".format(type(self.variable)))

        hidden_dims = self.hidden_dims + [self.output_dim]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=self.bias)]
            layers += [create_activation(self.activation_fn_cfg)]

        self._layers = nn.ModuleList(layers)


class VariableDecoder(BaseNetwork):
    def __init__(
        self,
        variable: Variable,
        input_dim: int = 100,
        hidden_dims: Optional[List[int]] = None,
        bias: bool = True,
        activation_fn_cfg: Optional[DictConfig] = None,
    ):
        self.variable = variable
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.bias = bias
        self.activation_fn_cfg = activation_fn_cfg

        self.name = "{}_decoder".format(variable.name)

        super(VariableDecoder, self).__init__()
        self._model_filename = "{}.pth".format(self.name)

    def build(self):
        layers = [create_activation(self.activation_fn_cfg)]

        hidden_dims = [self.input_dim] + self.hidden_dims
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=self.bias)]
            layers += [create_activation(self.activation_fn_cfg)]

        if len(self.hidden_dims) == 0:
            hidden_dim = self.input_dim
        else:
            hidden_dim = self.hidden_dims[-1]

        if isinstance(self.variable, ContinuousVariable):
            layers.append(nn.Linear(hidden_dim, self.variable.dim * 2))
        elif isinstance(self.variable, RadianVariable):
            layers.append(nn.Linear(hidden_dim, self.variable.dim * 2))
        elif isinstance(self.variable, DiscreteVariable):
            layers.append(nn.Linear(hidden_dim, self.variable.n))
            layers.append(nn.Softmax())
        elif isinstance(self.variable, BinaryVariable):
            layers.append(nn.Linear(hidden_dim, 1))
            layers.append(nn.Sigmoid())
        else:
            raise NotImplementedError("Type {} is not supported by VariableDecoder".format(type(self.variable)))

        self._layers = nn.ModuleList(layers)
