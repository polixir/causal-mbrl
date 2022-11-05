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

        self.learn_reward = reward_mech is None
        self.learn_termination = termination_mech is None

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
        # transition
        self.transition.learn(*self.get_loader(real_replay_buffer, "transition"))
        # reward-mech
        self.reward_mech.learn(*self.get_loader(real_replay_buffer, "reward_mech"))
        # termination-mech
        self.termination_mech.learn(*self.get_loader(real_replay_buffer, "termination_mech"))
