import abc
from collections import ChainMap
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from torch.utils.data import DataLoader
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.utils.variables import space2dict
from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech
from cmrl.models.data_loader import BufferDataset, EnsembleBufferDataset, collate_fn


class Dynamics:
    def __init__(
        self,
        transition: BaseCausalMech,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_mech: Optional[BaseCausalMech] = None,
        termination_mech: Optional[BaseCausalMech] = None,
        seed: int = 7,
        logger: Optional[Logger] = None,
    ):
        self.transition = transition
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_mech = reward_mech
        self.termination_mech = termination_mech
        self.seed = seed
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
            seed=self.seed,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        valid_dataset = BufferDataset(
            real_replay_buffer,
            self.observation_space,
            self.action_space,
            is_valid=True,
            mech=mech,
            repeat=self.transition.ensemble_num,
            seed=self.seed,
        )
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

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
            obs_dict = space2dict(batch_obs, self.observation_space, "obs", repeat=7, to_tensor=True)
            act_dict = space2dict(batch_action, self.action_space, "act", repeat=7, to_tensor=True)

            inputs = ChainMap(obs_dict, act_dict)
            outputs = self.transition.forward(inputs)

        info = {"origin-next_obs": torch.concat([tensor[:, :, :1] for tensor in outputs.values()], dim=-1).cpu().numpy()}

        return torch.concat([tensor.mean(dim=0)[:, :1] for tensor in outputs.values()], dim=-1).cpu().numpy(), None, None, info
