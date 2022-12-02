import os
from typing import Optional
from functools import partial

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
import wandb

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

        self.env, self.reward_fn, self.termination_fn, self.get_init_obs_fn = make_env(self.cfg)
        self.eval_env, *_ = make_env(self.cfg)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        self.logger = logger_configure("log", ["tensorboard", "multi_csv", "stdout"])

        if cfg.wandb:
            wandb.init(
                project="causal-mbrl",
                group=cfg.exp_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
            )

        # create ``cmrl`` dynamics
        self.dynamics = create_dynamics(self.cfg, self.env.observation_space, self.env.action_space, logger=self.logger)

        if not self.cfg.transition.discovery:
            self.dynamics.transition.set_oracle_graph(self.env.get_transition_graph())
        if self.cfg.reward_mech.learn and not self.cfg.reward_mech.discovery:
            self.dynamics.reward_mech.set_oracle_graph(self.env.get_reward_mech_graph())
        if self.cfg.termination_mech.learn and not self.cfg.termination_mech.discovery:
            self.dynamics.termination_mech.set_oracle_graph(self.env.get_termination_mech_graph())

        # create sb3's replay buffer for real offline data
        self.real_replay_buffer = ReplayBuffer(
            cfg.task.num_steps,
            self.env.observation_space,
            self.env.action_space,
            self.cfg.device,
            handle_timeout_termination=False,
        )

        self.partial_fake_env = partial(
            VecFakeEnv,
            self.cfg.algorithm.num_envs,
            self.env.observation_space,
            self.env.action_space,
            self.dynamics,
            self.reward_fn,
            self.termination_fn,
            self.get_init_obs_fn,
            self.real_replay_buffer,
            penalty_coeff=self.cfg.task.penalty_coeff,
            logger=self.logger,
        )
        self.agent = create_agent(self.cfg, self.fake_env, self.logger)

    @property
    def fake_env(self) -> VecFakeEnv:
        return self.partial_fake_env(
            deterministic=self.cfg.algorithm.deterministic,
            max_episode_steps=self.env.spec.max_episode_steps,
            branch_rollout=False,
        )

    @property
    def callback(self) -> BaseCallback:
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

        self.agent.learn(total_timesteps=self.cfg.task.num_steps, callback=self.callback)

    def _setup_learn(self):
        pass
