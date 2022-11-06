import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
from functools import partial

import gym.wrappers
from gym import spaces
import hydra
from hydra.utils import instantiate
import numpy as np
import omegaconf
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.models.dynamics import Dynamics
from cmrl.models.fake_env import VecFakeEnv
from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech
from cmrl.models.util import parse_space
from cmrl.types import DiscreteVariable, ContinuousVariable, BinaryVariable, InitObsFnType, RewardFnType, TermFnType


def create_agent(cfg, fake_env: VecFakeEnv, logger: Optional[Logger] = None):
    agent = instantiate(cfg.algorithm.agent)(env=fake_env)
    agent = cast(BaseAlgorithm, agent)
    agent.set_logger(logger)

    return agent


def create_dynamics(
    cfg,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    logger: Optional[Logger] = None,
):
    obs_variables = parse_space(observation_space, "obs")
    act_variables = parse_space(action_space, "act")
    next_obs_variables = parse_space(observation_space, "next_obs")

    # transition
    assert cfg.transition.learn
    # TODO: share encoders
    transition = instantiate(cfg.transition.mech)(
        input_variables=obs_variables + act_variables,
        output_variables=next_obs_variables,
        variable_encoders=None,
        variable_decoders=None,
        logger=logger,
    )
    transition = cast(BaseCausalMech, transition)
    # reward mech
    if cfg.reward_mech.learn:
        reward_mech = instantiate(cfg.reward_mech.mech)(
            input_variables=obs_variables + act_variables + next_obs_variables,
            output_variables=[ContinuousVariable("reward", dim=1, low=-np.inf, high=np.inf)],
            variable_encoders=None,
            variable_decoders=None,
            logger=logger,
        )
        reward_mech = cast(BaseCausalMech, reward_mech)
    else:
        reward_mech = None
    # termination mech
    if cfg.reward_mech.learn:
        termination_mech = instantiate(cfg.termination_mech.mech)(
            input_variables=obs_variables + act_variables + next_obs_variables,
            output_variables=[BinaryVariable("terminal")],
            variable_encoders=None,
            variable_decoders=None,
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


# def create_dynamics(
#     dynamics_cfg: omegaconf.DictConfig,
#     obs_shape: Tuple[int, ...],
#     act_shape: Tuple[int, ...],
#     logger: Optional[Logger] = None,
#     load_dir: Optional[Union[str, pathlib.Path]] = None,
#     load_device: Optional[str] = None,
# ):
#     if dynamics_cfg.name == "plain_dynamics":
#         dynamics_class = PlainEnsembleDynamics
#     elif dynamics_cfg.name == "constraint_based_dynamics":
#         dynamics_class = ConstraintBasedDynamics
#     else:
#         raise NotImplementedError
#
#     dynamics_cfg = get_complete_dynamics_cfg(dynamics_cfg, obs_shape, act_shape)
#     transition = hydra.utils.instantiate(dynamics_cfg.transition, _recursive_=False)
#     if dynamics_cfg.multi_step == "none":
#         pass
#     elif dynamics_cfg.multi_step.startswith("forward_euler"):
#         repeat_times = int(dynamics_cfg.multi_step[len("forward_euler") + 1 :])
#         transition = ForwardEulerTransition(transition, repeat_times)
#     else:
#         raise NotImplementedError
#
#     if dynamics_cfg.learned_reward:
#         reward_mech = hydra.utils.instantiate(dynamics_cfg.reward_mech, _recursive_=False)
#     else:
#         reward_mech = None
#
#     if dynamics_cfg.learned_termination:
#         termination_mech = hydra.utils.instantiate(dynamics_cfg.termination_mech, _recursive_=False)
#         raise NotImplementedError
#     else:
#         termination_mech = None
#
#     dynamics_model = dynamics_class(
#         transition=transition,
#         learned_reward=dynamics_cfg.learned_reward,
#         reward_mech=reward_mech,
#         learned_termination=dynamics_cfg.learned_termination,
#         termination_mech=termination_mech,
#         optim_lr=dynamics_cfg.optim_lr,
#         weight_decay=dynamics_cfg.weight_decay,
#         logger=logger,
#     )
#     if load_dir:
#         dynamics_model.load(load_dir, load_device)
#
#     return dynamics_model
