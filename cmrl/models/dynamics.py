import abc
from collections import ChainMap
import pathlib
from typing import Dict, List, Optional, Tuple, Union
from functools import partial

import numpy as np
import torch
from gym import spaces
from torch.utils.data import DataLoader
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.utils.variables import to_dict_by_space
from cmrl.models.causal_mech.base import BaseCausalMech
from cmrl.models.data_loader import buffer_to_dict
from cmrl.types import Obs2StateFnType


class Dynamics:
    def __init__(
            self,
            transition: BaseCausalMech,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            obs2state_fn: Obs2StateFnType,
            reward_mech: Optional[BaseCausalMech] = None,
            termination_mech: Optional[BaseCausalMech] = None,
            seed: int = 7,
            logger: Optional[Logger] = None,
    ):
        self.transition = transition
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs2state_fn = obs2state_fn
        self.reward_mech = reward_mech
        self.termination_mech = termination_mech
        self.seed = seed
        self.logger = logger

        self.learn_reward = reward_mech is not None
        self.learn_termination = termination_mech is not None

        self.device = self.transition.device
        pass

    def learn(self, real_replay_buffer: ReplayBuffer, work_dir: Optional[Union[str, pathlib.Path]] = None, **kwargs):
        get_dataset = partial(
            buffer_to_dict,
            observation_space=self.observation_space,
            action_space=self.action_space,
            obs2state_fn=self.obs2state_fn,
            replay_buffer=real_replay_buffer,
        )

        # transition
        self.transition.learn(*get_dataset(mech="transition"), work_dir=work_dir)
        # reward-mech
        if self.learn_reward:
            self.reward_mech.learn(*get_dataset(mech="reward_mech"), work_dir=work_dir)
        # termination-mech
        if self.learn_termination:
            self.termination_mech.learn(*get_dataset(mech="termination_mech"), work_dir=work_dir)

    def step(self, batch_obs, batch_action):
        with torch.no_grad():
            obs_dict = to_dict_by_space(batch_obs, self.observation_space, "obs", repeat=7, to_tensor=True)
            act_dict = to_dict_by_space(batch_action, self.action_space, "act", repeat=7, to_tensor=True)

            inputs = ChainMap(obs_dict, act_dict)
            outputs = self.transition.forward(inputs)

        info = {
            "origin-next_obs": torch.concat([tensor[:, :, :1] for tensor in outputs.values()], dim=-1).cpu().numpy()}

        return torch.concat([tensor.mean(dim=0)[:, :1] for tensor in outputs.values()],
                            dim=-1).cpu().numpy(), None, None, info

    # def set_oracle_graph(self, graph):
