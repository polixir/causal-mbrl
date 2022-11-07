from typing import List, Optional, Union, Dict

import numpy as np
import torch
from gym import spaces
from omegaconf import DictConfig

from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.utils.types import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable
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
    data: np.ndarray, space: spaces.Space, prefix="obs", repeat: Optional[int] = None, to_tensor: bool = False
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
            dict_data[name] = torch.from_numpy(dict_data[name])

    return dict_data


def dict2space(
    data: Dict[str, Union[np.ndarray, torch.Tensor]], space: spaces.Space
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    pass


def create_encoders(
    input_variables: List[Variable],
    node_dim: int,
    hidden_dims: Optional[List[int]] = None,
    bias: bool = True,
    activation_fn_cfg: Optional[DictConfig] = None,
    device: Union[str, torch.device] = "cpu",
):
    encoders = {}
    for var in input_variables:
        assert var.name not in encoders, "duplicate name in decoders: {}".format(var.name)
        encoders[var.name] = VariableEncoder(
            variable=var, output_dim=node_dim, hidden_dims=hidden_dims, bias=bias, activation_fn_cfg=activation_fn_cfg
        ).to(device)
    return encoders


def create_decoders(
    input_variables: List[Variable],
    node_dim: int,
    hidden_dims: Optional[List[int]] = None,
    bias: bool = True,
    activation_fn_cfg: Optional[DictConfig] = None,
    normal_distribution: bool = True,
    device: Union[str, torch.device] = "cpu",
):
    decoders = {}
    for var in input_variables:
        assert var.name not in decoders, "duplicate name in decoders: {}".format(var.name)
        decoders[var.name] = VariableDecoder(
            variable=var,
            input_dim=node_dim,
            hidden_dims=hidden_dims,
            bias=bias,
            activation_fn_cfg=activation_fn_cfg,
            normal_distribution=normal_distribution,
        ).to(device)
    return decoders


def load_offline_data(env, replay_buffer: ReplayBuffer, dataset_name: str, use_ratio: float = 1):
    assert hasattr(env, "get_dataset"), "env must have `get_dataset` method"

    data_dict = env.get_dataset(dataset_name)
    all_data_num = len(data_dict["observations"])
    sample_data_num = int(use_ratio * all_data_num)
    sample_idx = np.random.permutation(all_data_num)[:sample_data_num]

    assert replay_buffer.n_envs == 1
    assert replay_buffer.buffer_size >= sample_data_num

    if sample_data_num == replay_buffer.buffer_size:
        replay_buffer.full = True
        replay_buffer.pos = 0
    else:
        replay_buffer.pos = sample_data_num

    # set all data
    for attr in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
        getattr(replay_buffer, attr)[:sample_data_num, 0] = data_dict[attr][sample_idx]
