from typing import Optional, List

import numpy as np
import torch
from torch import nn as nn
from torch import Tensor
from itertools import product

from cmrl.models.util import truncated_normal_


# partial from https://github.com/phlippe/ENCO/blob/main/causal_discovery/multivariable_mlp.py
class ParallelLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        extra_dims: Optional[List[int]] = None,
        bias: bool = True,
        init_type: str = "truncated_normal",
    ):
        """Linear layer with the same properties as Parallel MLP. It effectively applies N independent linear layers
        in parallel.

        Args:
            input_dim: Number of input dimensions per layer.
            output_dim: Number of output dimensions per layer.
            extra_dims: Number of neural networks to have in parallel (e.g. number of variables). Can have multiple
                dimensions if needed.
            bias: Weather using bias in this layer.
            init_type: How to initialize weights and biases.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.extra_dims = [] if extra_dims is None else extra_dims
        self.init_type = init_type

        self.weight = nn.Parameter(torch.zeros(*self.extra_dims, self.input_dim, self.output_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(*self.extra_dims, 1, self.output_dim))
            self.use_bias = True
        else:
            self.use_bias = False

        self.init_params()

    def init_params(self):
        """Initialize weights and biases. Currently, only `kaiming_uniform` and `truncated_normal` are supported.

        Returns: None

        """
        if self.init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        elif self.init_type == "truncated_normal":
            stddev = 1 / (2 * np.sqrt(self.input_dim))
            for dims in product(*map(range, self.extra_dims)):
                truncated_normal_(self.weight.data[dims], std=stddev)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        xw = x.matmul(self.weight)
        if self.use_bias:
            return xw + self.bias
        else:
            return xw

    @property
    def device(self) -> torch.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        Returns: device
        """
        for param in self.parameters():
            return param.device
        return torch.device("cpu")

    def extra_repr(self):
        return 'input_dims={}, output_dims={}, extra_dims={}, bias={}, init_type="{}"'.format(
            self.input_dim, self.output_dim, str(self.extra_dims), self.use_bias, self.init_type
        )


class RadianLayer(nn.Module):
    def __init__(self) -> None:
        super(RadianLayer, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.remainder(input + torch.pi, 2 * torch.pi) - torch.pi
