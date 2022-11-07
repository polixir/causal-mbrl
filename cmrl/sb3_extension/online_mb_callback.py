import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from cmrl.models.fake_env import VecFakeEnv
from cmrl.models.dynamics import Dynamics


class OnlineModelBasedCallback(BaseCallback):
    def __init__(
        self,
        env: gym.Env,
        dynamics: Dynamics,
        real_replay_buffer: ReplayBuffer,
        # online RL
        total_online_timesteps: int = int(1e5),
        initial_exploration_steps: int = 1000,
        freq_train_model: int = 250,
        # dynamics learning
        longest_epoch: int = -1,
        improvement_threshold: float = 0.01,
        patience: int = 5,
        work_dir: Optional[Union[str, pathlib.Path]] = None,
        device: str = "cpu",
    ):
        super(OnlineModelBasedCallback, self).__init__(verbose=2)

        self.env = DummyVecEnv([lambda: env])
        self.dynamics = dynamics
        self.real_replay_buffer = real_replay_buffer
        # online RL
        self.total_online_timesteps = total_online_timesteps
        self.initial_exploration_steps = initial_exploration_steps
        self.freq_train_model = freq_train_model
        # dynamics learning
        self.longest_epoch = longest_epoch
        self.improvement_threshold = improvement_threshold
        self.patience = patience
        self.work_dir = work_dir
        self.device = device

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.now_online_timesteps = 0
        self._last_obs = None

    def _on_step(self) -> bool:
        if self.n_calls % self.freq_train_model == 0:
            # dump some residual log before dynamics learn
            self.model.logger.dump(step=self.num_timesteps)

            self.dynamics.learn(
                self.real_replay_buffer,
                longest_epoch=self.longest_epoch,
                improvement_threshold=self.improvement_threshold,
                patience=self.patience,
                work_dir=self.work_dir,
            )

            self.step_and_add(explore=False)

        if self.now_online_timesteps >= self.total_online_timesteps:
            return False
        return True

    def _on_training_start(self):
        assert self.env.num_envs == 1

        self._last_obs = self.env.reset()
        while self.now_online_timesteps < self.initial_exploration_steps:
            self.step_and_add(explore=True)

    def step_and_add(self, explore=True):
        if explore:
            actions = np.array([self.action_space.sample()])
        else:
            actions, _ = self.model.predict(self._last_obs, deterministic=False)
        buffer_actions = self.model.policy.scale_action(actions)

        new_obs, rewards, dones, infos = self.env.step(actions)
        self.now_online_timesteps += 1

        next_obs = deepcopy(new_obs)
        if dones[0] and infos[0].get("terminal_observation") is not None:
            next_obs[0] = infos[0]["terminal_observation"]
        self.real_replay_buffer.add(self._last_obs, next_obs, buffer_actions, rewards, dones, infos)

        self._last_obs = new_obs.copy()
