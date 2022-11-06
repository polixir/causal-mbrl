from typing import Dict, Optional, Tuple, cast

import emei
import gym
import omegaconf

import cmrl.utils.types


def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def get_term_and_reward_fn(
    cfg: omegaconf.DictConfig,
) -> Tuple[cmrl.utils.types.TermFnType, Optional[cmrl.utils.types.RewardFnType]]:
    return None, None


def make_env(
    cfg: omegaconf.DictConfig,
) -> Tuple[
    emei.EmeiEnv,
    cmrl.utils.types.TermFnType,
    Optional[cmrl.utils.types.RewardFnType],
    Optional[cmrl.utils.types.InitObsFnType],
]:
    if "gym___" in cfg.task.env:
        env = gym.make(cfg.task.env.split("___")[1])
        term_fn, reward_fn = get_term_and_reward_fn(cfg)
        init_obs_fn = None
    elif "emei___" in cfg.task.env:
        env_name, params, = cfg.task.env.split(
            "___"
        )[1:3]
        kwargs = dict([(item.split("=")[0], to_num(item.split("=")[1])) for item in params.split("&")])
        env = cast(emei.EmeiEnv, gym.make(env_name, **kwargs))
        reward_fn = env.get_reward
        term_fn = env.get_terminal
        init_obs_fn = env.get_batch_init_obs
    else:
        raise NotImplementedError

    # set seed
    env.reset(seed=cfg.seed)
    env.observation_space.seed(cfg.seed + 1)
    env.action_space.seed(cfg.seed + 2)
    return env, reward_fn, term_fn, init_obs_fn
