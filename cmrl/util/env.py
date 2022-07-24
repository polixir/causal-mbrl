import gym
import omegaconf
from typing import Dict, Optional, Tuple, Union, cast
import torch
import emei
import cmrl.types


def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def get_term_and_reward_fn(cfg: omegaconf.DictConfig) -> Tuple[cmrl.types.TermFnType,
                                                               Optional[cmrl.types.RewardFnType]]:
    return None, None


def make_env(cfg: omegaconf.DictConfig) -> Tuple[emei.EmeiEnv,
                                                 cmrl.types.TermFnType,
                                                 Optional[cmrl.types.RewardFnType]]:
    if "gym___" in cfg.task.env:
        env = gym.make(cfg.task.env.split("___")[1])
        term_fn, reward_fn = get_term_and_reward_fn(cfg)
    elif "emei___" in cfg.task.env:
        env_name, params, = cfg.task.env.split("___")[1:3]
        kwargs = dict([(item.split("=")[0], to_num(item.split("=")[1])) for item in params.split("&")])
        env = cast(emei.EmeiEnv, gym.make(env_name, **kwargs))
        term_fn = env.get_terminal_by_next_obs
        reward_fn = env.get_reward_by_next_obs
    else:
        raise NotImplementedError
    return env, term_fn, reward_fn
