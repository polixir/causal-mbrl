import gym
from gym import spaces
import torch
from torch.utils.data import Dataset, default_collate
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer


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
        if isinstance(self.replay_buffer, DictReplayBuffer):
            # TODO: DictReplayBuffer case
            raise NotImplementedError
        else:
            observations = self.replay_buffer.observations[: self.size, 0].astype(np.float32)
            assert len(observations.shape) == 2
            next_observations = self.replay_buffer.next_observations[: self.size, 0].astype(np.float32)

            observations_dict = dict([("obs_{}".format(i), obs[:, None]) for i, obs in enumerate(observations.T)])
            next_observations_dict = dict(
                [("next_obs_{}".format(i), obs[:, None]) for i, obs in enumerate(next_observations.T)]
            )

        assert isinstance(self.observation_space, spaces.Box)
        # TODO: other spaces for observation and action(e.g. one-hot for spaces.Discrete)
        # see: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py#L85

        actions = self.replay_buffer.actions[: self.size, 0]
        actions_dict = dict([("act_{}".format(i), obs[:, None]) for i, obs in enumerate(actions.T)])

        self.inputs = {}
        self.inputs.update(observations_dict)
        self.inputs.update(actions_dict)

        if self.mech == "transition":
            self.outputs = next_observations_dict
        elif self.mech == "reward_mech":
            rewards = self.replay_buffer.rewards[: self.size, 0]
            rewards_dict = {"reward": rewards[:, None]}
            self.inputs.update(next_observations_dict)
            self.outputs = rewards_dict
        else:
            dones = self.replay_buffer.dones[: self.size, 0]
            timeouts = self.replay_buffer.timeouts[: self.size, 0]
            terminals_dict = {"terminal": (dones * (1 - timeouts))[:, None]}
            self.inputs.update(next_observations_dict)
            self.outputs = terminals_dict

    def __getitem__(self, item):
        index = self.indexes[item]

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
        only_for_training: bool = True,
        seed: int = 10086,
    ):
        self.ensemble_num = ensemble_num
        self.only_for_training = only_for_training

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

        if self.only_for_training:  # call ``np.random`` ensemble-num + 1 times
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
