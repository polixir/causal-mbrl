import argparse
import os
import pathlib
from typing import Generator, List, Optional, Tuple, cast, Union

import cmrl
import cmrl.models
import cmrl.agent
from cmrl.diagnostics.util import load_hydra_cfg
from cmrl.util.env import make_env


class Runner:
    def __init__(
            self,
            agent_dir: str,
            checkpoint: Optional[str],
    ):
        self.agent_dir = agent_dir
        self.cfg = load_hydra_cfg(self.agent_dir)
        self.env, *_ = make_env(self.cfg)
        self.agent = cmrl.agent.load_agent(self.agent_dir, self.env, ckpt=checkpoint)

    def run(self):
        # from emei.util import random_policy_test
        obs = self.env.reset()
        self.env.render()
        total_reward = 0
        while True:
            action = self.agent.act(obs)
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
        "--checkpoint",
        type=str,
        default=None,
        help="The directory where the agent configuration and data is stored. "
             "If not provided, a random agent will be used.",
    )
    args = parser.parse_args()

    runner = Runner(
        agent_dir=args.agent_dir,
        checkpoint=args.checkpoint
    )
    runner.run()
