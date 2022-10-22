import numpy as np
import torch
from torch import nn as nn

import cmrl.models.util as model_util


def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""

    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        model_util.truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)
    elif isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            model_util.truncated_normal_(m.weight.data[i], std=stddev)
        m.bias.data.fill_(0.0)
    elif isinstance(m, ParallelEnsembleLinearLayer):
        num_parallel, num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_parallel):
            for j in range(num_members):
                model_util.truncated_normal_(m.weight.data[i, j], std=stddev)
        m.bias.data.fill_(0.0)


class EnsembleLinearLayer(nn.Module):
    """Implements an ensemble of layers.

    Args:
        in_size (int): the input size of this layer.
        out_size (int): the output size of this layer.
        use_bias (bool): use bias in this layer or not.
        ensemble_num (int): the ensemble dimension of this layer,
            the corresponding part of each dimension is called a "member".
    """

    def __init__(
            self,
            in_size: int,
            out_size: int,
            use_bias: bool = True,
            ensemble_num: int = 1,
    ):
        super().__init__()
        self.ensemble_num = ensemble_num
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.rand(self.ensemble_num, self.in_size, self.out_size))
        if use_bias:
            self.bias = nn.Parameter(torch.rand(self.ensemble_num, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x):
        xw = x.matmul(self.weight)
        if self.use_bias:
            return xw + self.bias
        else:
            return xw

    def __repr__(self) -> str:
        return (
            f"in_size={self.in_size}, out_size={self.out_size}, use_bias={self.use_bias}, " f"ensemble_num={self.ensemble_num}"
        )


class ParallelEnsembleLinearLayer(nn.Module):
    """Implements an ensemble of parallel layers.

    Args:
        in_size (int): the input size of this layer.
        out_size (int): the output size of this layer.
        use_bias (bool): use bias in this layer or not.
        parallel_num (int): the parallel dimension of this layer,
            the corresponding part of each dimension is called a "sub-network".
        ensemble_num (int): the ensemble dimension of this layer,
            the corresponding part of each dimension is called a "member".
    """

    def __init__(
            self,
            in_size: int,
            out_size: int,
            use_bias: bool = True,
            parallel_num: int = 1,
            ensemble_num: int = 1,
    ):
        super().__init__()
        self.parallel_num = parallel_num
        self.ensemble_num = ensemble_num
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.rand(self.parallel_num, self.ensemble_num, self.in_size, self.out_size))
        if use_bias:
            self.bias = nn.Parameter(torch.rand(self.parallel_num, self.ensemble_num, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x):
        xw = x.matmul(self.weight)
        if self.use_bias:
            return xw + self.bias
        else:
            return xw

    def __repr__(self) -> str:
        return (
            f"in_size={self.in_size}, out_size={self.out_size}, use_bias={self.use_bias}, "
            f"parallel_num={self.parallel_num}, ensemble_num={self.ensemble_num}"
        )
