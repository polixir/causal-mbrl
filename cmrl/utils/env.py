from typing import Dict, Optional, Tuple, cast

import numpy as np
import emei
import gym
import omegaconf
from stable_baselines3.common.buffers import ReplayBuffer

import cmrl.utils.variables
from cmrl.types import TermFnType, RewardFnType, InitObsFnType, Obs2StateFnType


def make_env(
        cfg: omegaconf.DictConfig,
) -> Tuple[emei.EmeiEnv, tuple]:
    env = cast(emei.EmeiEnv, gym.make(cfg.task.env_id, **cfg.task.params))
    fns = (
        env.get_batch_reward,
        env.get_batch_terminal,
        env.get_batch_init_obs,
        env.obs2state,
        env.state2obs
    )

    # set seed
    env.reset(seed=cfg.seed)
    env.state_space.seed(cfg.seed + 1)
    env.action_space.seed(cfg.seed + 2)
    return env, fns


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
        # if attr == "dones" and attr not in data_dict and "terminals" in data_dict:
        #     replay_buffer.dones[:sample_data_num, 0] = data_dict["terminals"][sample_idx]
        #     continue
        getattr(replay_buffer, attr)[:sample_data_num, 0] = data_dict[attr][sample_idx]

    for attr in ["extra_obs", "next_extra_obs"]:
        setattr(
            replay_buffer,
            attr,
            np.zeros((replay_buffer.buffer_size, replay_buffer.n_envs) + data_dict[attr].shape[1:], dtype=np.float32)
        )
        getattr(replay_buffer, attr)[:sample_data_num, 0] = data_dict[attr][sample_idx]
