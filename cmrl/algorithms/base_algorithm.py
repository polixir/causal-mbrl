import os
from typing import Optional
from functools import partial

import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

from cmrl.models.util import load_offline_data
from cmrl.models.fake_env import VecFakeEnv
from cmrl.sb3_extension.logger import configure as logger_configure
from cmrl.sb3_extension.eval_callback import EvalCallback
from cmrl.utils.creator import create_dynamics, create_agent
from cmrl.utils.env import make_env


class BaseAlgorithm:
    def __init__(
        self,
        cfg: DictConfig,
        work_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        self.work_dir = work_dir or os.getcwd()

        self.env, self.reward_fn, self.termination_fn, self.get_init_obs_fn = make_env(cfg)
        self.eval_env, *_ = make_env(cfg)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.logger = logger_configure("log", ["tensorboard", "multi_csv", "stdout"])

        # create ``cmrl`` dynamics
        self.dynamics = create_dynamics(cfg, self.env.observation_space, self.env.action_space, logger=self.logger)

        # create sb3's replay buffer for real offline data
        self.real_replay_buffer = ReplayBuffer(
            cfg.task.num_steps, self.env.observation_space, self.env.action_space, cfg.device, handle_timeout_termination=False
        )

        self.partial_fake_env = partial(
            VecFakeEnv,
            cfg.algorithm.num_envs,
            self.env.observation_space,
            self.env.action_space,
            self.dynamics,
            self.reward_fn,
            self.termination_fn,
            self.get_init_obs_fn,
            self.real_replay_buffer,
            penalty_coeff=cfg.task.penalty_coeff,
            logger=self.logger,
        )

        self.fake_env = self.get_fake_env()

        self.agent = create_agent(cfg, self.fake_env, self.logger)

        self.callback = self.get_callback()

    def get_fake_env(self) -> VecFakeEnv:
        return self.partial_fake_env(
            deterministic=self.cfg.algorithm.deterministic,
            max_episode_steps=self.env.spec.max_episode_steps,
            branch_rollout=False,
        )

    def get_callback(self) -> BaseCallback:
        fake_eval_env = self.partial_fake_env(
            deterministic=True, max_episode_steps=self.env.spec.max_episode_steps, branch_rollout=False
        )
        return EvalCallback(
            self.eval_env,
            fake_eval_env,
            n_eval_episodes=self.cfg.task.n_eval_episodes,
            best_model_save_path="./",
            eval_freq=1000,
            deterministic=True,
            render=False,
        )

    def learn(self):
        self._setup_learn()

        self.dynamics.learn(
            real_replay_buffer=self.real_replay_buffer,
            longest_epoch=self.cfg.task.longest_epoch,
            improvement_threshold=self.cfg.task.improvement_threshold,
            patience=self.cfg.task.patience,
            work_dir=self.work_dir,
        )

        self.agent.learn(total_timesteps=self.cfg.task.num_steps, callback=self.callback)

    def _setup_learn(self):
        load_offline_data(self.env, self.real_replay_buffer, self.cfg.task.dataset, self.cfg.task.use_ratio)
