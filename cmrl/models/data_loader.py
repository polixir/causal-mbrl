from typing import Optional, MutableMapping

from gym import spaces, Env
import torch
from torch.utils.data import Dataset, default_collate
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer

from cmrl.utils.variables import to_dict_by_space


def buffer_to_dict(state_space, action_space, obs2state_fn, replay_buffer: ReplayBuffer, mech: str, device: str = "cpu"):
    assert mech in ["transition", "reward_mech", "termination_mech"]
    # dict action is not supported by SB3(so not done by cmrl)
    assert not isinstance(action_space, spaces.Dict)
    assert hasattr(replay_buffer, "extra_obs")
    assert hasattr(replay_buffer, "next_extra_obs")

    real_buffer_size = replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos

    if hasattr(replay_buffer, "extra_obs"):
        states = obs2state_fn(replay_buffer.observations[:real_buffer_size, 0], replay_buffer.extra_obs[:real_buffer_size, 0])
    else:
        states = replay_buffer.observations[:real_buffer_size, 0]
    state_dict = to_dict_by_space(states, state_space, prefix="obs", to_tensor=True)
    act_dict = to_dict_by_space(replay_buffer.actions[:real_buffer_size, 0], action_space, prefix="act", to_tensor=True)

    if hasattr(replay_buffer, "next_extra_obs"):
        next_states = obs2state_fn(
            replay_buffer.next_observations[:real_buffer_size, 0], replay_buffer.next_extra_obs[:real_buffer_size, 0]
        )
    else:
        next_states = replay_buffer.next_observations[:real_buffer_size, 0]
    next_state_dict = to_dict_by_space(next_states, state_space, prefix="next_obs", to_tensor=True)

    inputs = {}
    inputs.update(state_dict)
    inputs.update(act_dict)

    if mech == "transition":
        outputs = next_state_dict
    elif mech == "reward_mech":
        rewards = replay_buffer.rewards[:real_buffer_size, 0]
        rewards_dict = {"reward": torch.from_numpy(rewards[:, None])}
        inputs.update(next_state_dict)
        outputs = rewards_dict
    elif mech == "termination_mech":
        terminals = replay_buffer.dones[:real_buffer_size, 0] * (1 - replay_buffer.timeouts[:real_buffer_size, 0])
        terminals_dict = {"terminal": torch.from_numpy(terminals[:, None])}
        inputs.update(next_state_dict)
        outputs = terminals_dict
    else:
        raise NotImplementedError("support mechs in [transition, reward_mech, termination_mech] only")

    return inputs, outputs


class EnsembleBufferDataset(Dataset):
    def __init__(
        self,
        inputs: MutableMapping,
        outputs: MutableMapping,
        training: bool = False,
        train_ratio: float = 0.8,
        ensemble_num: int = 7,
        seed: int = 10086,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.training = training
        self.train_ratio = train_ratio
        self.ensemble_num = ensemble_num
        self.seed = seed
        self.indexes = None

        size = next(iter(inputs.values())).shape[0]

        np.random.seed(self.seed)
        permutation = np.random.permutation(size)
        if self.training:
            train_indexes = permutation[: int(size * self.train_ratio)]
            indexes = [np.random.permutation(train_indexes) for _ in range(self.ensemble_num)]
        else:
            valid_indexes = permutation[int(size * self.train_ratio) :]
            indexes = [valid_indexes for _ in range(self.ensemble_num)]
        self.indexes = np.array(indexes).T

    def __getitem__(self, item):
        index = self.indexes[item]

        inputs = dict([(key, self.inputs[key][index]) for key in self.inputs])
        outputs = dict([(key, self.outputs[key][index]) for key in self.outputs])
        return inputs, outputs

    def __len__(self):
        return len(self.indexes)


def collate_fn(data):
    inputs, outputs = default_collate(data)
    inputs = dict([(key, value.transpose(0, 1)) for key, value in inputs.items()])
    outputs = dict([(key, value.transpose(0, 1)) for key, value in outputs.items()])
    return [inputs, outputs]
