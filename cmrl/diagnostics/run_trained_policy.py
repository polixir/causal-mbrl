import argparse
import os
import pathlib
from typing import Generator, List, Optional, Tuple, cast, Union
import math

import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm

import cmrl
import cmrl.models
import cmrl.agent
from cmrl.util.config import load_hydra_cfg
from cmrl.util.env import make_env


class Runner:
    def __init__(
            self,
            agent_dir: str,
            type: str = "best",
            device="cuda:0"
    ):
        self.agent_dir = pathlib.Path(agent_dir)
        self.cfg = load_hydra_cfg(self.agent_dir)
        self.cfg.device = device
        self.env, *_ = make_env(self.cfg)

        agent_cfg = self.cfg.algorithm.agent
        agent_class: BaseAlgorithm = eval(agent_cfg._target_)
        self.agent = agent_class.load(self.agent_dir / "best_model")

    def run(self):
        # from emei.util import random_policy_test
        obs = self.env.reset()
        self.env.render()
        total_reward = 0
        while True:
            action, state = self.agent.predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.env.render()
            if done:
                print(total_reward)
                total_reward = 0
                obs = self.env.reset()
            else:
                total_reward += reward
                obs = next_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "agent_dir",
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
        agent_dir=args.agent_dir,
        type=args.type
    )
    runner.run()
