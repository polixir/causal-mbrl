import argparse
import math
import os
import pathlib
from copy import copy, deepcopy
from typing import Generator, List, Optional, Tuple, Union, cast

import gym
import hydra
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback

import cmrl
import cmrl.agent
import cmrl.models
import cmrl.utils.creator
from cmrl.utils.config import load_hydra_cfg
from cmrl.utils.env import make_env


class Runner:
    def __init__(self, model_dir: str, device: str = "cuda:0", render: bool = False):
        self.render = render

        self.model_path = pathlib.Path(model_dir)
        self.cfg = load_hydra_cfg(self.model_path)
        self.cfg.device = device
        self.env, self.term_fn, self.reward_fn, self.init_obs_fn = make_env(self.cfg)
        cmrl.agent.complete_agent_cfg(self.env, self.cfg.algorithm.agent)

        self.dynamics = cmrl.util.creator.create_dynamics(
            self.cfg.dynamics,
            self.env.state_space.shape,
            self.env.action_space.shape,
            load_dir=self.model_path,
            load_device=device,
        )

        numpy_generator = np.random.default_rng(seed=self.cfg.seed)

        eval_env_cfg = deepcopy(self.cfg.algorithm.agent.env)
        eval_env_cfg.num_envs = 1
        fake_eval_env = cast(cmrl.models.VecFakeEnv, hydra.utils.instantiate(eval_env_cfg))
        fake_eval_env.set_up(
            self.dynamics,
            self.reward_fn,
            self.term_fn,
            self.init_obs_fn,
            max_episode_steps=self.env.spec.max_episode_steps,
            penalty_coeff=0.3,
        )
        fake_eval_env.seed(seed=self.cfg.seed)
        self.fake_eval_env = cmrl.models.GymBehaviouralFakeEnv(fake_eval_env, self.env)

        agent_cfg = self.cfg.algorithm.agent
        agent_class: BaseAlgorithm = eval(agent_cfg._target_)
        self.agent = agent_class.load(self.model_path / "best_model")

    def run(self):
        # from emei.utils import random_policy_test
        obs = self.fake_eval_env.reset()
        if self.render:
            self.fake_eval_env.render()
        episode_reward = 0
        episode_length = 0
        while True:
            # action = self.gym_fake_env.action_space.sample()
            action, state = self.agent.predict(obs)
            next_obs, reward, done, _ = self.fake_eval_env.step(action)

            if self.render:
                self.fake_eval_env.render()
            if done:
                print(episode_reward, episode_length)
                episode_reward = 0
                episode_length = 0
                obs = self.fake_eval_env.reset()
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
    parser.add_argument(
        "--render",
        action="store_true",
    )
    args = parser.parse_args()

    runner = Runner(model_dir=args.model_dir, render=args.render)
    runner.run()
