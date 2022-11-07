from typing import Optional

from gym import spaces
import torch
from torch.utils.data import Dataset, default_collate
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer

from cmrl.models.util import space2dict


class BufferDataset(Dataset):
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        mech: str,
        is_valid: bool = False,
        train_ratio: float = 0.8,
        seed: int = 10086,
        repeat: Optional[int] = None,
    ):
        assert mech in ["transition", "reward_mech", "termination_mech"]
        # dict action is not supported by SB3(so not done by cmrl)
        assert not isinstance(action_space, spaces.Dict)

        self.replay_buffer = replay_buffer
        self.observation_space = observation_space
        self.action_space = action_space
        self.mech = mech
        self.is_valid = is_valid
        self.train_ratio = train_ratio
        self.seed = seed
        self.repeat = repeat

        if self.repeat:
            assert self.repeat > 1, "repeat must be a int greater than 1"

        self.size = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos

        self.inputs = None
        self.outputs = None
        self.load_from_buffer()

        self.indexes = None
        self.build_indexes()

    def build_indexes(self):
        np.random.seed(self.seed)
        permutation = np.random.permutation(self.size)
        if self.is_valid:  # for valid set
            self.indexes = permutation[int(self.size * self.train_ratio) :]
        else:  # for train set
            self.indexes = permutation[: int(self.size * self.train_ratio)]

    def load_from_buffer(self):
        obs_dict = space2dict(
            self.replay_buffer.observations[: self.size, 0],
            self.observation_space,
            prefix="obs",
            to_tensor=True,
        )
        act_dict = space2dict(
            self.replay_buffer.actions[: self.size, 0],
            self.action_space,
            prefix="act",
            to_tensor=True,
        )
        next_obs_dict = space2dict(
            self.replay_buffer.next_observations[: self.size, 0],
            self.observation_space,
            prefix="next_obs",
            to_tensor=True,
            # device=self.device
        )

        self.inputs = {}
        self.inputs.update(obs_dict)
        self.inputs.update(act_dict)

        if self.mech == "transition":
            self.outputs = next_obs_dict
        elif self.mech == "reward_mech":
            rewards = self.replay_buffer.rewards[: self.size, 0]
            rewards_dict = {"reward": torch.from_numpy(rewards[:, None])}
            self.inputs.update(next_obs_dict)
            self.outputs = rewards_dict
        else:
            terminals = self.replay_buffer.dones[: self.size, 0] * (1 - self.replay_buffer.timeouts[: self.size, 0])
            terminals_dict = {"terminal": torch.from_numpy(terminals[:, None])}
            self.inputs.update(next_obs_dict)
            self.outputs = terminals_dict

    def __getitem__(self, item):
        index = self.indexes[item]
        if self.repeat:
            assert len(self.indexes.shape) == 1, "repeating conflicts with ensemble"
            index = np.tile(index, self.repeat)

        inputs = dict([(key, self.inputs[key][index]) for key in self.inputs])
        outputs = dict([(key, self.outputs[key][index]) for key in self.outputs])
        return inputs, outputs

    def __len__(self):
        return len(self.indexes)


class EnsembleBufferDataset(BufferDataset):
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        mech: str,
        is_valid: bool = False,
        train_ratio: float = 0.8,
        ensemble_num: int = 7,
        train_ensemble: bool = True,
        seed: int = 10086,
    ):
        self.ensemble_num = ensemble_num
        self.train_ensemble = train_ensemble

        super(EnsembleBufferDataset, self).__init__(
            replay_buffer=replay_buffer,
            observation_space=observation_space,
            action_space=action_space,
            mech=mech,
            is_valid=is_valid,
            train_ratio=train_ratio,
            seed=seed,
        )

    def build_indexes(self):
        indexes = []
        np.random.seed(self.seed)

        if self.train_ensemble:  # call ``np.random`` ensemble-num + 1 times
            assert not self.is_valid
            train_indexes = np.random.permutation(self.size)[: int(self.size * self.train_ratio)]
            indexes = [np.random.permutation(train_indexes) for _ in range(self.ensemble_num)]
        else:
            for i in range(self.ensemble_num):
                permutation = np.random.permutation(self.size)
                if self.is_valid:
                    indexes.append(permutation[int(self.size * self.train_ratio) :])
                else:
                    indexes.append(permutation[: int(self.size * self.train_ratio)])
        self.indexes = np.array(indexes).T


def collate_fn(data):
    inputs, outputs = default_collate(data)
    inputs = dict([(key, value.transpose(0, 1)) for key, value in inputs.items()])
    outputs = dict([(key, value.transpose(0, 1)) for key, value in outputs.items()])
    return [inputs, outputs]
