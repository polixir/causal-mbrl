import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization,
)

from cmrl.models import VecFakeEnv


class OnlineModelBasedCallback(BaseCallback):
    def __init__(self, env, dynamics, num_steps: int = int(1e5), initial_exploration_steps: int = 1000, device: str = "cpu"):
        super(OnlineModelBasedCallback, self).__init__(verbose=2)

        self.env = DummyVecEnv([lambda: env])
        self.dynamics = dynamics
        self.num_steps = num_steps
        self.initial_exploration_steps = initial_exploration_steps
        self.device = device

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.replay_buffer = ReplayBuffer(
            num_steps,
            env.observation_space,
            env.action_space,
            device=self.device,
            n_envs=1,
            optimize_memory_usage=False,
        )

        self.num_timesteps = 0
        self._last_obs = None

    def _on_step(self) -> bool:
        pass

    def _on_rollout_start(self) -> None:
        self.step_and_add(explore=False)

    def _on_training_start(self):
        assert self.env.num_envs == 1

        self._last_obs = self.env.reset()
        while self.num_timesteps < self.initial_exploration_steps:
            self.step_and_add(explore=True)

    def step_and_add(self, explore=True):
        if explore:
            actions = np.array([self.action_space.sample()])
        else:
            actions, _ = self.model.predict(self._last_obs, deterministic=False)
        buffer_actions = self.model.policy.scale_action(actions)

        new_obs, rewards, dones, infos = self.env.step(actions)
        self.num_timesteps += 1

        next_obs = deepcopy(new_obs)
        if dones[0] and infos[0].get("terminal_observation") is not None:
            next_obs[0] = infos[0]["terminal_observation"]
        self.replay_buffer.add(self._last_obs, next_obs, buffer_actions, rewards, dones, infos)

        self._last_obs = new_obs.copy()
