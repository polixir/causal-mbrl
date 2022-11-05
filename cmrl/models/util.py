# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from omegaconf import DictConfig

import cmrl.types
from cmrl.types import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder


# inplace truncated normal function for pytorch.
# credit to https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(mean, std, size=(bound_violations,), device=tensor.device)
    return tensor


def to_tensor(x: cmrl.types.TensorType):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise ValueError("Input must be torch.Tensor or np.ndarray.")


def parse_space(space: spaces.Space, prefix="obs") -> List[Variable]:
    variables = []
    if isinstance(space, spaces.Box):
        for i, (low, high) in enumerate(zip(space.low, space.high)):
            variables.append(ContinuousVariable(dim=1, low=low, high=high, name="{}_{}".format(prefix, i)))
    elif isinstance(space, spaces.Discrete):
        variables.append(DiscreteVariable(n=space.n, name="{}_0".format(prefix)))
    elif isinstance(space, spaces.MultiDiscrete):
        for i, n in enumerate(space.nvec):
            variables.append(DiscreteVariable(n=n, name="{}_{}".format(prefix, i)))
    elif isinstance(space, spaces.MultiBinary):
        for i in range(space.n):
            variables.append(BinaryVariable(name="{}_{}".format(prefix, i)))
    elif isinstance(space, spaces.Dict):
        # TODO
        raise NotImplementedError

    return variables


def create_encoders(
    input_variables: List[Variable],
    node_dim: int,
    hidden_dims: Optional[List[int]] = None,
    bias: bool = True,
    activation_fn_cfg: Optional[DictConfig] = None,
):
    encoders = {}
    for var in input_variables:
        assert var.name not in encoders, "Duplicate name in decoders: {}".format(var.name)
        encoders[var.name] = VariableEncoder(
            variable=var, node_dim=node_dim, hidden_dims=hidden_dims, bias=bias, activation_fn_cfg=activation_fn_cfg
        )
    return encoders


def create_decoders(
    input_variables: List[Variable],
    node_dim: int,
    hidden_dims: Optional[List[int]] = None,
    bias: bool = True,
    activation_fn_cfg: Optional[DictConfig] = None,
    normal_distribution: bool = True,
):
    decoders = {}
    for var in input_variables:
        assert var.name not in decoders, "Duplicate name in decoders: {}".format(var.name)
        decoders[var.name] = VariableDecoder(
            variable=var,
            node_dim=node_dim,
            hidden_dims=hidden_dims,
            bias=bias,
            activation_fn_cfg=activation_fn_cfg,
            normal_distribution=normal_distribution,
        )
    return decoders
