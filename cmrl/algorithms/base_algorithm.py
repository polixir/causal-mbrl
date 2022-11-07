import os
from typing import Optional
from functools import partial

import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

from cmrl.models.fake_env import VecFakeEnv
from cmrl.sb3_extension.logger import configure as logger_configure
from cmrl.sb3_extension.eval_callback import EvalCallback
from cmrl.utils.creator import create_dynamics, create_agent
from cmrl.utils.env import make_env


def load_offline_data(env, replay_buffer: ReplayBuffer, dataset_name: str, use_ratio: float = 1):
    assert hasattr(env, "get_dataset"), "env must have `get_dataset` method"

    data_dict = env.get_dataset(dataset_name)
    all_data_num = len(data_dict["observations"])
    sample_data_num = int(use_ratio * all_data_num)
    sample_idx = np.random.permutation(all_data_num)[:sample_data_num]

    assert replay_buffer.n_envs == 1
    assert replay_buffer.buffer_size >= sample_data_num

    if sample_data_num == replay_buffer.buffer_size:
        replay_buffer.full = True
        replay_buffer.pos = 0
    else:
        replay_buffer.pos = sample_data_num

    # set all data
    for attr in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
        getattr(replay_buffer, attr)[:sample_data_num, 0] = data_dict[attr][sample_idx]


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

        # create ``cmrl`` dynamics
        self.dynamics = create_dynamics(self.cfg, self.env.observation_space, self.env.action_space, logger=self.logger)

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

        self.fake_env = self.get_fake_env()

        self.agent = create_agent(self.cfg, self.fake_env, self.logger)

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
