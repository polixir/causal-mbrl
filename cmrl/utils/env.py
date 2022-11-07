from typing import Dict, Optional, Tuple, cast

import numpy as np
import emei
import gym
import omegaconf
from stable_baselines3.common.buffers import ReplayBuffer

import cmrl.utils.variables
from cmrl.types import TermFnType, RewardFnType, InitObsFnType


def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def get_term_and_reward_fn(
    cfg: omegaconf.DictConfig,
) -> Tuple[Optional[TermFnType], Optional[RewardFnType]]:
    return None, None


def make_env(
    cfg: omegaconf.DictConfig,
) -> Tuple[emei.EmeiEnv, TermFnType, Optional[RewardFnType], Optional[InitObsFnType],]:
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


def load_offline_data(env, replay_buffer: ReplayBuffer, dataset_name: str, use_ratio: float = 1):
    assert hasattr(env, "get_dataset"), "env must have `get_dataset` method"

    data_dict = env.get_dataset(dataset_name)
    all_data_num = len(data_dict["observations"])
    sample_data_num = int(use_ratio * all_data_num)
    sample_idx = np.random.permutation(all_data_num)[:sample_data_num]

    assert replay_buffer.n_envs == 1
    assert replay_buffer.buffer_size >= sample_data_num

    if sample_data_num == replay_buffer.buffer_size:
        replay_buffer.full = True
        replay_buffer.pos = 0
    else:
        replay_buffer.pos = sample_data_num

    # set all data
    for attr in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
        getattr(replay_buffer, attr)[:sample_data_num, 0] = data_dict[attr][sample_idx]
