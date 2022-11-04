from typing import Union
from abc import abstractmethod

import torch
from gym import Space


class BaseCausalMechanism:
    def __init__(self, obs_space: Space, action_space: Space, deterministic: bool):
        self.obs_space = obs_space
        self.action_space = action_space
        self.deterministic = deterministic

        self.network = None
        self.graph = None
        pass

    @abstractmethod
    def predict(self, obs, action, next_obs=None):
        pass
