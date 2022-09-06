import argparse
import os
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


class GymBehaviouralFakeEnv(gym.Env):
    def __init__(self,
                 fake_env,
                 real_env):
        self.fake_env = fake_env
        self.real_env = real_env

        self.action_space = self.real_env.action_space
        self.observation_space = self.real_env.observation_space

    def step(self, action):
        batch_next_obs, batch_reward, batch_terminal = self.fake_env.step(np.array([action], dtype=np.float32),
                                                                          deterministic=True)
        self.real_env.set_state_by_obs(batch_next_obs[0])
        return batch_next_obs[0], batch_reward[0][0], batch_terminal[0][0], False, {}

    def render(self, mode="human"):
        assert mode == "human"
        self.real_env.render()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        obs = self.real_env.reset()
        self.fake_env.reset(np.array([obs], dtype=np.float32))
        return obs


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
        self.gym_fake_env = GymBehaviouralFakeEnv(fake_env=self.fake_env,
                                                  real_env=self.env)

        self.agent = cmrl.agent.load_agent(self.model_path, self.env, type=type, device=device)

    def run(self):
        # from emei.util import random_policy_test
        obs = self.gym_fake_env.reset()
        self.gym_fake_env.render()
        total_reward = 0
        while True:
            # action = self.gym_fake_env.action_space.sample()
            action = self.agent.act(obs)
            next_obs, reward, terminal, truncated, _ = self.gym_fake_env.step(action)
            self.gym_fake_env.render()
            if terminal or truncated:
                print(total_reward)
                total_reward = 0
                obs = self.gym_fake_env.reset()
            else:
                total_reward += reward
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
    runner.run()
