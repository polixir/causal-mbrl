from dataclasses import dataclass
from typing import Optional, Dict, Union, List

from gym import spaces
import numpy as np
import torch


@dataclass
class Variable:
    name: str
    pass


@dataclass
class ContinuousVariable(Variable):
    dim: int
    low: np.ndarray = None
    high: np.ndarray = None


@dataclass
class BinaryVariable(Variable):
    pass


@dataclass
class DiscreteVariable(Variable):
    n: int


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


def space2dict(
    data: np.ndarray,
    space: spaces.Space,
    prefix="obs",
    repeat: Optional[int] = None,
    to_tensor: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    if repeat:
        assert repeat > 1, "repeat must be a int greater than 1"

    dict_data = {}
    if isinstance(space, spaces.Box):  # shape: (batch-size, node-num), every node has exactly one dim
        for i, (low, high) in enumerate(zip(space.low, space.high)):
            # shape: (batch-size, specific-dim)
            dict_data["{}_{}".format(prefix, i)] = data[:, i, None].astype(np.float32)
    else:
        # TODO
        raise NotImplementedError

    for name in dict_data:
        if repeat:
            # shape: (repeat-dim, batch-size, specific-dim)
            dict_data[name] = np.tile(dict_data[name][None, :, :], [repeat, 1, 1])
        if to_tensor:
            dict_data[name] = torch.from_numpy(dict_data[name]).to(device)

    return dict_data


def dict2space(
    data: Dict[str, Union[np.ndarray, torch.Tensor]], space: spaces.Space
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    pass
