import abc
import pathlib
from typing import Any, Union, Optional

import gym
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import omegaconf

import cmrl.models
import cmrl.types


class Agent:
    """Abstract class for all agents."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def reset(self):
        pass


class RandomAgent(Agent):
    """An agent that samples action from the environments action space.

    Args:
        env (gym.Env): the environment on which the agent will act.
    """

    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return self.env.action_space.sample()


def complete_agent_cfg(
        env: gym.Env, agent_cfg: omegaconf.DictConfig
):
    """Completes an agent's configuration given information from the environment.

    The goal of this function is to completed information about state and action shapes and ranges,
    without requiring the user to manually enter this into the Omegaconf configuration object.

    It will check for and complete any of the following keys:

        - "obs_dim": set to env.observation_space.shape
        - "action_dim": set to env.action_space.shape
        - "action_range": set to max(env.action_space.high) - min(env.action_space.low)
        - "action_lb": set to env.action_space.low
        - "action_ub": set to env.action_space.high

    Note:
        If the user provides any of these values in the Omegaconf configuration object, these
        *will not* be overridden by this function.

    """
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    def _check_and_replace(key: str, value: Any, cfg: omegaconf.DictConfig):
        if key in cfg.keys() and key not in cfg:
            setattr(cfg, key, value)

    # create numpy object by existed object
    def _create_numpy_config(array):
        return {
            "_target_": "numpy.array",
            "object": array.tolist(),
            "dtype": str(array.dtype)
        }

    _check_and_replace("num_inputs", obs_shape[0], agent_cfg)
    if "action_space" in agent_cfg.keys() and isinstance(
            agent_cfg.action_space, omegaconf.DictConfig
    ):
        _check_and_replace("low", _create_numpy_config(env.action_space.low), agent_cfg.action_space)
        _check_and_replace(
            "high", _create_numpy_config(env.action_space.high), agent_cfg.action_space
        )
        _check_and_replace("shape", env.action_space.shape, agent_cfg.action_space)

    if "obs_dim" in agent_cfg.keys() and "obs_dim" not in agent_cfg:
        agent_cfg.obs_dim = obs_shape[0]
    if "action_dim" in agent_cfg.keys() and "action_dim" not in agent_cfg:
        agent_cfg.action_dim = act_shape[0]
    if "action_range" in agent_cfg.keys() and "action_range" not in agent_cfg:
        agent_cfg.action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max()),
        ]
    if "action_lb" in agent_cfg.keys() and "action_lb" not in agent_cfg:
        agent_cfg.action_lb = _create_numpy_config(env.action_space.low)
    if "action_ub" in agent_cfg.keys() and "action_ub" not in agent_cfg:
        agent_cfg.action_ub = _create_numpy_config(env.action_space.high)

    return agent_cfg


def load_agent(agent_path: Union[str, pathlib.Path],
               env: gym.Env,
               ckpt: Optional[str] = None,
               device: Optional[str] = None) -> Agent:
    """Loads an agent from a Hydra config file at the given path.

    For agent of type "pytorch_sac.agent.sac.SACAgent", the directory
    must contain the following files:

        - ".hydra/config.yaml": the Hydra configuration for the agent.
        - "critic.pth": the saved checkpoint for the critic.
        - "actor.pth": the saved checkpoint for the actor.

    Args:
        agent_path (str or pathlib.Path): a path to the directory where the agent is saved.
        env (gym.Env): the environment on which the agent will operate (only used to complete
            the agent's configuration).

    Returns:
        (Agent): the new agent.
    """
    agent_path = pathlib.Path(agent_path)
    cfg = omegaconf.OmegaConf.load(agent_path / ".hydra" / "config.yaml")
    cfg.device = device

    if cfg.algorithm.agent._target_ == "cmrl.third_party.pytorch_sac.sac.SAC":
        import cmrl.third_party.pytorch_sac as pytorch_sac

        from .sac_wrapper import SACAgent

        complete_agent_cfg(env, cfg.algorithm.agent)
        agent: pytorch_sac.SAC = hydra.utils.instantiate(cfg.algorithm.agent)
        if ckpt is not None:
            agent.load_checkpoint(ckpt_path=agent_path / "sac_checkpoints" / "checkpoint_{}.pth".format(ckpt),
                                  device=device)
        else:
            agent.load_checkpoint(ckpt_path=agent_path / "sac_best.pth", device=device)
        return SACAgent(agent)
    else:
        raise ValueError("Invalid agent configuration.")
