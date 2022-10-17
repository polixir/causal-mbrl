import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gym.wrappers
import hydra
import numpy as np
import omegaconf
from stable_baselines3.common.logger import Logger

from cmrl.models.dynamics import ConstraintBasedDynamics, PlainEnsembleDynamics
from cmrl.models.transition import ForwardEulerTransition
from cmrl.util.config import get_complete_dynamics_cfg


def create_dynamics(
    dynamics_cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    logger: Optional[Logger] = None,
    load_dir: Optional[Union[str, pathlib.Path]] = None,
    load_device: Optional[str] = None,
):
    if dynamics_cfg.name == "plain_dynamics":
        dynamics_class = PlainEnsembleDynamics
    elif dynamics_cfg.name == "constraint_based_dynamics":
        dynamics_class = ConstraintBasedDynamics
    else:
        raise NotImplementedError

    dynamics_cfg = get_complete_dynamics_cfg(dynamics_cfg, obs_shape, act_shape)
    transition = hydra.utils.instantiate(dynamics_cfg.transition, _recursive_=False)
    if dynamics_cfg.multi_step == "none":
        pass
    elif dynamics_cfg.multi_step.startswith("forward_euler"):
        repeat_times = int(dynamics_cfg.multi_step[len("forward_euler") + 1 :])
        transition = ForwardEulerTransition(transition, repeat_times)
    else:
        raise NotImplementedError

    if dynamics_cfg.learned_reward:
        reward_mech = hydra.utils.instantiate(dynamics_cfg.reward_mech, _recursive_=False)
    else:
        reward_mech = None

    if dynamics_cfg.learned_termination:
        termination_mech = hydra.utils.instantiate(dynamics_cfg.termination_mech, _recursive_=False)
        raise NotImplementedError
    else:
        termination_mech = None

    dynamics_model = dynamics_class(
        transition=transition,
        learned_reward=dynamics_cfg.learned_reward,
        reward_mech=reward_mech,
        learned_termination=dynamics_cfg.learned_termination,
        termination_mech=termination_mech,
        optim_lr=dynamics_cfg.optim_lr,
        weight_decay=dynamics_cfg.weight_decay,
        logger=logger,
    )
    if load_dir:
        dynamics_model.load(load_dir, load_device)

    return dynamics_model
