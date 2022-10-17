import pathlib
from typing import Optional, cast
from copy import deepcopy

import emei
import hydra
import numpy as np
from omegaconf import DictConfig

from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.types import InitObsFnType, RewardFnType, TermFnType
from cmrl.models.dynamics import BaseDynamics
from cmrl.util.config import get_complete_dynamics_cfg, load_hydra_cfg
from cmrl.models.fake_env import VecFakeEnv


def is_same_dict(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            return False
        else:
            if isinstance(dict1[key], DictConfig) and isinstance(dict2[key], DictConfig):
                if not is_same_dict(dict1[key], dict2[key]):
                    return False
            else:
                if dict1[key] != dict2[key]:
                    return False
    return True


def maybe_load_trained_offline_model(dynamics: BaseDynamics, cfg, obs_shape, act_shape, work_dir):
    work_dir = pathlib.Path(work_dir)
    if "." not in work_dir.name:  # exp by hydra's MULTIRUN mode
        task_exp_dir = work_dir.parent.parent.parent
    else:
        task_exp_dir = work_dir.parent.parent
    dynamics_cfg = cfg.dynamics

    for date_dir in task_exp_dir.glob(r"*"):
        for time_dir in date_dir.glob(r"*"):
            if (time_dir / "multirun.yaml").exists():  # exp by hydra's MULTIRUN mode, multi exp in this time
                this_time_exp_dir_list = list(time_dir.glob(r"*"))
            else:  # only one exp in this time
                this_time_exp_dir_list = [time_dir]

            for exp_dir in this_time_exp_dir_list:
                if not (exp_dir / ".hydra").exists():
                    continue
                exp_cfg = load_hydra_cfg(exp_dir)
                exp_dynamics_cfg = get_complete_dynamics_cfg(exp_cfg.dynamics, obs_shape, act_shape)

                if exp_cfg.seed == cfg.seed and is_same_dict(dynamics_cfg, exp_dynamics_cfg):
                    exist_model_file = True
                    for mech in dynamics.learn_mech:
                        mech_file_name = getattr(dynamics, mech).model_file_name
                        if not (exp_dir / mech_file_name).exists():
                            exist_model_file = False
                    if exist_model_file:
                        dynamics.load(exp_dir)
                        print("loaded dynamics from {}".format(exp_dir))
                        return True
    return False


def setup_fake_env(
    cfg: DictConfig,
    agent: BaseAlgorithm,
    dynamics,
    reward_fn: Optional[RewardFnType],
    termination_fn: Optional[TermFnType],
    get_init_obs_fn: Optional[InitObsFnType],
    real_replay_buffer: Optional[ReplayBuffer] = None,
    logger=None,
    max_episode_steps: int = 1000,
    penalty_coeff: Optional[float] = 0,
):
    fake_env = cast(VecFakeEnv, agent.env)
    fake_env.set_up(
        dynamics,
        reward_fn,
        termination_fn,
        get_init_obs_fn=get_init_obs_fn,
        real_replay_buffer=real_replay_buffer,
        logger=logger,
        max_episode_steps=max_episode_steps,
        penalty_coeff=penalty_coeff,
    )
    agent.env = VecMonitor(fake_env)

    fake_eval_env_cfg = deepcopy(cfg.algorithm.agent.env)
    fake_eval_env_cfg.num_envs = cfg.task.n_eval_episodes
    fake_eval_env = cast(VecFakeEnv, hydra.utils.instantiate(fake_eval_env_cfg))
    fake_eval_env.set_up(
        dynamics,
        reward_fn,
        termination_fn,
        get_init_obs_fn=get_init_obs_fn,
        max_episode_steps=max_episode_steps,
        penalty_coeff=penalty_coeff,
    )
    fake_eval_env.seed(seed=cfg.seed)
    return fake_eval_env


def load_offline_data(cfg: DictConfig, env, replay_buffer: ReplayBuffer):
    assert hasattr(env, "get_dataset"), "env must have `get_dataset` method"

    params, dataset_type = cfg.task.env.split("___")[-2:]
    data_dict = env.get_dataset("{}-{}".format(params, dataset_type))
    all_data_num = len(data_dict["observations"])
    sample_data_num = int(cfg.task.use_ratio * all_data_num)
    sample_idx = np.random.permutation(all_data_num)[:sample_data_num]

    replay_buffer.extend(
        data_dict["observations"][sample_idx],
        data_dict["next_observations"][sample_idx],
        data_dict["actions"][sample_idx],
        data_dict["rewards"][sample_idx],
        data_dict["terminals"][sample_idx].astype(bool) | data_dict["timeouts"][sample_idx].astype(bool),
        [{}] * sample_data_num,
    )
