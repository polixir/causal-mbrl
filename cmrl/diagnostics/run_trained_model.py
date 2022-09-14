import argparse
import os
import torch
import pathlib
from typing import Generator, List, Optional, Tuple, cast, Union
import math
import gym
import numpy as np
import cmrl
import cmrl.models
import cmrl.agent
from cmrl.util.config import load_hydra_cfg
from cmrl.util.env import make_env
import cmrl.util.creator

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


class Runner:
    def __init__(
            self,
            model_dir: str,
            type: str = "best",
            device="cuda:0"
    ):
        self.model_path = pathlib.Path(model_dir)
        self.cfg = load_hydra_cfg(self.model_path)
        self.cfg.device = device
        self.env, self.term_fn, self.reward_fn = make_env(self.cfg)

        self.dynamics = cmrl.util.creator.create_dynamics(self.cfg.dynamics,
                                                          self.env.observation_space.shape,
                                                          self.env.action_space.shape,
                                                          load_dir=self.model_path,
                                                          load_device=device)

        numpy_generator = np.random.default_rng(seed=self.cfg.seed)
        self.fake_env = cmrl.models.FakeEnv(self.env,
                                            self.dynamics,
                                            self.reward_fn,
                                            self.term_fn,
                                            generator=numpy_generator,
                                            penalty_coeff=self.cfg.algorithm.penalty_coeff)
        self.gym_fake_env = gym.wrappers.TimeLimit(cmrl.models.GymBehaviouralFakeEnv(fake_env=self.fake_env,
                                                                                     real_env=self.env),
                                                   max_episode_steps=self.env.spec.max_episode_steps,
                                                   new_step_api=False)
        test_env, *_ = make_env(self.cfg)
        self.test_env = gym.wrappers.TimeLimit(test_env,
                                               max_episode_steps=self.env.spec.max_episode_steps,
                                               new_step_api=False)

        self.agent = cmrl.agent.load_agent(self.model_path, self.env, type=type, device=device)

    def train_policy(self):
        model = PPO("MlpPolicy", self.gym_fake_env, verbose=1)
        log_path = str(self.model_path / "diagnostics" / "ppo" / "log")
        tb_path = str(self.model_path / "diagnostics" / "ppo" / "tb")
        eval_callback = EvalCallback(self.test_env, best_model_save_path=log_path,
                                     log_path=log_path, eval_freq=1000,
                                     deterministic=True, render=False)

        model.learn(total_timesteps=int(1e6), callback=eval_callback, tb_log_name=tb_path)
        model.save(self.model_path / "diagnostics" / "ppo")

    def run(self):
        # from emei.util import random_policy_test
        obs = self.gym_fake_env.reset()
        self.gym_fake_env.render()
        episode_reward = 0
        episode_length = 0
        while True:
            # action = self.gym_fake_env.action_space.sample()
            action = self.agent.act(obs)
            next_obs, reward, terminal, truncated, _ = self.gym_fake_env.step(action)
            self.gym_fake_env.render()
            if terminal or truncated:
                print(episode_reward, episode_length)
                episode_reward = 0
                episode_length = 0
                obs = self.gym_fake_env.reset()
            else:
                episode_reward += reward
                episode_length += 1
                obs = next_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=str,
        help="The directory where the agent configuration and data is stored. "
             "If not provided, a random agent will be used.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="best",
    )
    args = parser.parse_args()

    runner = Runner(
        model_dir=args.model_dir,
        type=args.type
    )
    # runner.run()
    runner.train_policy()
