from typing import Optional, cast

from gym import spaces
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.base_class import BaseAlgorithm

from cmrl.models.dynamics import Dynamics
from cmrl.models.fake_env import VecFakeEnv
from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech
from cmrl.models.util import parse_space
from cmrl.utils.types import ContinuousVariable, BinaryVariable


def create_agent(cfg: DictConfig, fake_env: VecFakeEnv, logger: Optional[Logger] = None):
    agent = instantiate(cfg.algorithm.agent)(env=fake_env)
    agent = cast(BaseAlgorithm, agent)
    agent.set_logger(logger)

    return agent


def create_dynamics(
    cfg: DictConfig,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    logger: Optional[Logger] = None,
):
    obs_variables = parse_space(observation_space, "obs")
    act_variables = parse_space(action_space, "act")
    next_obs_variables = parse_space(observation_space, "next_obs")

    # transition
    assert cfg.transition.learn, "transition must be learned, or you should try model-free RL:)"
    transition = instantiate(cfg.transition.mech)(
        input_variables=obs_variables + act_variables,
        output_variables=next_obs_variables,
        logger=logger,
    )
    transition = cast(BaseCausalMech, transition)

    # reward mech
    assert cfg.reward_mech.mech.multi_step == "none", "reward-mech must be one-step"
    if cfg.reward_mech.learn:
        reward_mech = instantiate(cfg.reward_mech.mech)(
            input_variables=obs_variables + act_variables + next_obs_variables,
            output_variables=[ContinuousVariable("reward", dim=1, low=-np.inf, high=np.inf)],
            logger=logger,
        )
        reward_mech = cast(BaseCausalMech, reward_mech)
    else:
        reward_mech = None

    # termination mech
    assert cfg.termination_mech.mech.multi_step == "none", "termination-mech must be one-step"
    if cfg.termination_mech.learn:
        termination_mech = instantiate(cfg.termination_mech.mech)(
            input_variables=obs_variables + act_variables + next_obs_variables,
            output_variables=[BinaryVariable("terminal")],
            logger=logger,
        )
        termination_mech = cast(BaseCausalMech, termination_mech)
    else:
        termination_mech = None

    dynamics = Dynamics(
        transition=transition,
        reward_mech=reward_mech,
        termination_mech=termination_mech,
        observation_space=observation_space,
        action_space=action_space,
        logger=logger,
    )

    return dynamics
