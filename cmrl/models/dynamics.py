import abc
import collections
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from torch.utils.data import DataLoader
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.models.reward_mech.base_reward_mech import BaseRewardMech
from cmrl.models.termination_mech.base_termination_mech import BaseTerminationMech
from cmrl.models.transition.base_transition import BaseTransition
from cmrl.types import InteractionBatch
from cmrl.util.transition_iterator import BootstrapIterator, TransitionIterator
from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech
from cmrl.models.data_loader import BufferDataset, EnsembleBufferDataset, collate_fn


class Dynamics:
    def __init__(
        self,
        transition: BaseCausalMech,
        reward_mech: Optional[BaseCausalMech],
        termination_mech: Optional[BaseCausalMech],
        observation_space: spaces.Space,
        action_space: spaces.Space,
        logger: Optional[Logger] = None,
    ):
        self.transition = transition
        self.reward_mech = reward_mech
        self.termination_mech = termination_mech
        self.observation_space = observation_space
        self.action_space = action_space
        self.logger = logger

        self.learn_reward = reward_mech is not None
        self.learn_termination = termination_mech is not None

        self.device = self.transition.device
        pass

    def get_loader(self, real_replay_buffer, mech: str, batch_size: int = 128):
        train_dataset = EnsembleBufferDataset(
            real_replay_buffer,
            self.observation_space,
            self.action_space,
            is_valid=False,
            mech=mech,
            train_ensemble=True,
            ensemble_num=self.transition.ensemble_num,
        )
        train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)
        valid_dataset = BufferDataset(
            real_replay_buffer,
            self.observation_space,
            self.action_space,
            is_valid=True,
            mech=mech,
            repeat=self.transition.ensemble_num,
        )
        valid_loader = DataLoader(valid_dataset, batch_size=8, collate_fn=collate_fn)

        return train_loader, valid_loader

    def learn(
        self,
        real_replay_buffer: ReplayBuffer,
        # model learning
        longest_epoch: int = -1,
        improvement_threshold: float = 0.01,
        patience: int = 5,
        work_dir: Optional[Union[str, pathlib.Path]] = None,
        **kwargs
    ):
        longest_epoch = 0

        # transition
        self.transition.learn(
            *self.get_loader(real_replay_buffer, "transition"),
            longest_epoch=longest_epoch,
            improvement_threshold=improvement_threshold,
            patience=patience,
            work_dir=work_dir
        )
        # reward-mech
        if self.learn_reward:
            self.reward_mech.learn(
                *self.get_loader(real_replay_buffer, "reward_mech"),
                longest_epoch=longest_epoch,
                improvement_threshold=improvement_threshold,
                patience=patience,
                work_dir=work_dir
            )
        # termination-mech
        if self.learn_termination:
            self.termination_mech.learn(
                *self.get_loader(real_replay_buffer, "termination_mech"),
                longest_epoch=longest_epoch,
                improvement_threshold=improvement_threshold,
                patience=patience,
                work_dir=work_dir
            )

    def step(self, batch_obs, batch_action):
        with torch.no_grad():
            if isinstance(self.observation_space, spaces.Box):
                observations_dict = dict(
                    [
                        (
                            "obs_{}".format(i),
                            torch.from_numpy(np.tile(batch_obs.T[0][None, :, None], [7, 1, 1])).to(torch.float32),
                        )
                        for i, obs in enumerate(batch_obs.T)
                    ]
                )
            else:
                raise NotImplementedError

            if isinstance(self.action_space, spaces.Box):
                actions_dict = dict(
                    [
                        (
                            "act_{}".format(i),
                            torch.from_numpy(np.tile(batch_obs.T[0][None, :, None], [7, 1, 1])).to(torch.float32),
                        )
                        for i, obs in enumerate(batch_action.T)
                    ]
                )
            else:
                raise NotImplementedError

            inputs = {}
            inputs.update(observations_dict)
            inputs.update(actions_dict)
            outputs = self.transition.forward(inputs)

        info = {"origin-next_obs": torch.concat([tensor[:, :, :1] for tensor in outputs.values()], dim=-1).cpu().numpy()}

        return torch.concat([tensor.mean(dim=0)[:, :1] for tensor in outputs.values()], dim=-1).cpu().numpy(), None, None, info
