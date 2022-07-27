import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gym.wrappers
import hydra
import numpy as np
import omegaconf

from cmrl.util.replay_buffer import ReplayBuffer
from cmrl.models.dynamics.plain_dynamics import PlainEnsembleDynamics
from cmrl.util.logger import Logger


def create_replay_buffer(
        cfg: omegaconf.DictConfig,
        obs_shape: Sequence[int],
        act_shape: Sequence[int],
        obs_type: Type = np.float32,
        action_type: Type = np.float32,
        reward_type: Type = np.float32,
        load_dir: Optional[Union[str, pathlib.Path]] = None,
        collect_trajectories: bool = False,
        rng: Optional[np.random.Generator] = None,
) -> ReplayBuffer:
    """Creates a replay buffer from a given configuration.

    The configuration should be structured as follows::

        -cfg
          -algorithm
            -dataset_size (int, optional): the maximum size of the train dataset/buffer
          -overrides
            -num_steps (int, optional): how many steps to take in the environment
            -trial_length (int, optional): the maximum length for trials. Only needed if
                ``collect_trajectories == True``.

    The size of the replay buffer can be determined by either providing
    ``cfg.algorithm.dataset_size``, or providing ``cfg.overrides.num_steps``.
    Specifying dataset set size directly takes precedence over number of steps.

    Args:
        cfg (omegaconf.DictConfig): the configuration to use.
        obs_shape (Sequence of ints): the shape of observation arrays.
        act_shape (Sequence of ints): the shape of action arrays.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
        load_dir (optional str or pathlib.Path): if provided, the function will attempt to
            populate the buffers from "load_dir/replay_buffer.npz".
        collect_trajectories (bool, optional): if ``True`` sets the replay buffers to collect
            trajectory information. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Returns:
        (:class:`cmrl.replay_buffer.ReplayBuffer`): the replay buffer.
    """
    dataset_size = (
        cfg.algorithm.get("dataset_size", None) if "algorithm" in cfg else None
    )
    if not dataset_size:
        dataset_size = cfg.overrides.num_steps
    maybe_max_trajectory_len = None
    if collect_trajectories:
        if cfg.overrides.trial_length is None:
            raise ValueError(
                "cfg.overrides.trial_length must be set when "
                "collect_trajectories==True."
            )
        maybe_max_trajectory_len = cfg.overrides.trial_length

    replay_buffer = ReplayBuffer(
        dataset_size,
        obs_shape,
        act_shape,
        obs_type=obs_type,
        action_type=action_type,
        reward_type=reward_type,
        rng=rng,
        max_trajectory_length=maybe_max_trajectory_len,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        replay_buffer.load(str(load_dir))

    return replay_buffer


def get_complete_cfg(dynamics_cfg: omegaconf.DictConfig,
                     obs_shape: Tuple[int, ...],
                     act_shape: Tuple[int, ...], ):
    transition_cfg = dynamics_cfg.transition
    transition_cfg.obs_size = obs_shape[0]
    transition_cfg.action_size = act_shape[0]

    reward_cfg = dynamics_cfg.reward_mech
    reward_cfg.obs_size = obs_shape[0]
    reward_cfg.action_size = act_shape[0]

    termination_cfg = dynamics_cfg.termination_mech
    termination_cfg.obs_size = obs_shape[0]
    termination_cfg.action_size = act_shape[0]
    return transition_cfg, reward_cfg, termination_cfg


def create_dynamics(dynamics_cfg: omegaconf.DictConfig,
                    obs_shape: Tuple[int, ...],
                    act_shape: Tuple[int, ...],
                    logger: Optional[Logger] = None,
                    model_dir: Optional[Union[str, pathlib.Path]] = None):
    transition_cfg, reward_cfg, termination_cfg = get_complete_cfg(dynamics_cfg, obs_shape, act_shape)
    transition = hydra.utils.instantiate(transition_cfg, _recursive_=False)

    if dynamics_cfg.learned_reward:
        reward_mech = hydra.utils.instantiate(reward_cfg, _recursive_=False)
    else:
        reward_mech = None

    if dynamics_cfg.learned_termination:
        termination_mech = hydra.utils.instantiate(termination_cfg, _recursive_=False)
        raise NotImplementedError
    else:
        termination_mech = None

    dynamics_model = PlainEnsembleDynamics(
        transition=transition,
        learned_reward=dynamics_cfg.learned_reward,
        reward_mech=reward_mech,
        learned_termination=dynamics_cfg.learned_termination,
        termination_mech=termination_mech,
        optim_lr=dynamics_cfg.optim_lr,
        weight_decay=dynamics_cfg.weight_decay,
        logger=logger
    )
    if model_dir:
        dynamics_model.load(model_dir)

    return dynamics_model