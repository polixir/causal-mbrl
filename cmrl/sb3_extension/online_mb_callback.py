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
from cmrl.models.dynamics import BaseDynamics


class OnlineModelBasedCallback(BaseCallback):
    def __init__(
        self,
        env: gym.Env,
        dynamics: BaseDynamics,
        total_num_steps: int = int(1e5),
        initial_exploration_steps: int = 1000,
        freq_train_model: int = 250,
        device: str = "cpu",
    ):
        super(OnlineModelBasedCallback, self).__init__(verbose=2)

        self.env = DummyVecEnv([lambda: env])
        self.dynamics = dynamics
        self.total_num_steps = total_num_steps
        self.initial_exploration_steps = initial_exploration_steps
        self.freq_train_model = freq_train_model
        self.device = device

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.replay_buffer = ReplayBuffer(
            total_num_steps,
            env.observation_space,
            env.action_space,
            device=self.device,
            n_envs=1,
            optimize_memory_usage=False,
        )

        self.now_num_steps = 0
        self.step_times = 0
        self._last_obs = None

    def _on_step(self) -> bool:
        if self.step_times % self.freq_train_model == 0:
            self.dynamics.learn(self.replay_buffer)

        self.step_and_add(explore=False)
        self.step_times += 1

        if self.now_num_steps >= self.total_num_steps:
            return False
        return True

    def _on_training_start(self):
        assert self.env.num_envs == 1

        self._last_obs = self.env.reset()
        while self.now_num_steps < self.initial_exploration_steps:
            self.step_and_add(explore=True)

    def step_and_add(self, explore=True):
        if explore:
            actions = np.array([self.action_space.sample()])
        else:
            actions, _ = self.model.predict(self._last_obs, deterministic=False)
        buffer_actions = self.model.policy.scale_action(actions)

        new_obs, rewards, dones, infos = self.env.step(actions)
        self.now_num_steps += 1

        next_obs = deepcopy(new_obs)
        if dones[0] and infos[0].get("terminal_observation") is not None:
            next_obs[0] = infos[0]["terminal_observation"]
        self.replay_buffer.add(self._last_obs, next_obs, buffer_actions, rewards, dones, infos)

        self._last_obs = new_obs.copy()
