from typing import List, Optional, Union, Dict

import numpy as np
import torch
from gym import spaces

from cmrl.utils.types import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable


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
