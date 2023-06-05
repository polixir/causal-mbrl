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
class RadianVariable(Variable):
    dim: int


@dataclass
class BinaryVariable(Variable):
    pass


@dataclass
class DiscreteVariable(Variable):
    n: int


def parse_space(
        space: spaces.Space,
        prefix="obs",
        extra_info=None
) -> List[Variable]:
    extra_info = extra_info if extra_info is not None else {}

    variables = []
    if isinstance(space, spaces.Box):
        for i, (low, high) in enumerate(zip(space.low, space.high)):
            name = "{}_{}".format(prefix, i)
            if "Radian" in extra_info and name in extra_info["Radian"]:
                variables.append(RadianVariable(dim=1, name=name))
            else:
                variables.append(ContinuousVariable(dim=1, low=low, high=high, name=name))
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


def to_dict_by_space(
        data: np.ndarray,
        space: spaces.Space,
        prefix="obs",
        repeat: Optional[int] = None,
        to_tensor: bool = False,
        device: str = "cpu"
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Transform the interaction data from its own type to python's dict, by the signature of space.

    Args:
        data: interaction data from replay buffer
        space: space of gym
        prefix: prefix of the key in dict
        repeat: copy data in a new dimension
        to_tensor: transform the data from numpy's ndarray to torch's tensor
        device: device


    Returns: interaction data organized in dictionary form

    """
    if repeat:
        assert repeat > 1, "repeat must be a int greater than 1"

    dict_data = {}
    if isinstance(space, spaces.Box):
        # shape of data: (batch-size, node-num), every node has exactly one dim
        for i, (low, high) in enumerate(zip(space.low, space.high)):
            # shape of dict_data['xxx']: (batch-size, 1)
            dict_data["{}_{}".format(prefix, i)] = data[:, i, None].astype(np.float32)
    else:
        # TODO
        raise NotImplementedError

    for name in dict_data:
        if repeat:
            # shape of dict_data['xxx']: (repeat-dim, batch-size, specific-dim)
            # specific-dim is 1 for the case of spaces.Box
            dict_data[name] = np.tile(dict_data[name][None, :, :], [repeat, 1, 1])
        if to_tensor:
            dict_data[name] = torch.from_numpy(dict_data[name]).to(device)

    return dict_data


def dict2space(
        data: Dict[str, Union[np.ndarray, torch.Tensor]], space: spaces.Space
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    pass
