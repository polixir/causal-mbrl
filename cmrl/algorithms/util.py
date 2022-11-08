from typing import Optional, cast
from copy import deepcopy

import hydra
from omegaconf import DictConfig

from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.types import InitObsFnType, RewardFnType, TermFnType

# from cmrl.models.dynamics import BaseDynamics
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


# def maybe_load_trained_offline_model(dynamics: BaseDynamics, cfg, obs_shape, act_shape, work_dir):
#     work_dir = pathlib.Path(work_dir)
#     if "." not in work_dir.name:  # exp by hydra's MULTIRUN mode
#         task_exp_dir = work_dir.parent.parent.parent
#     else:
#         task_exp_dir = work_dir.parent.parent
#     dynamics_cfg = cfg.dynamics
#
#     for date_dir in task_exp_dir.glob(r"*"):
#         for time_dir in date_dir.glob(r"*"):
#             if (time_dir / "multirun.yaml").exists():  # exp by hydra's MULTIRUN mode, multi exp in this time
#                 this_time_exp_dir_list = list(time_dir.glob(r"*"))
#             else:  # only one exp in this time
#                 this_time_exp_dir_list = [time_dir]
#
#             for exp_dir in this_time_exp_dir_list:
#                 if not (exp_dir / ".hydra").exists():
#                     continue
#                 exp_cfg = load_hydra_cfg(exp_dir)
#                 exp_dynamics_cfg = get_complete_dynamics_cfg(exp_cfg.dynamics, obs_shape, act_shape)
#
#                 if exp_cfg.seed == cfg.seed and is_same_dict(dynamics_cfg, exp_dynamics_cfg):
#                     exist_model_file = True
#                     for mech in dynamics.learn_mech:
#                         mech_file_name = getattr(dynamics, mech).model_file_name
#                         if not (exp_dir / mech_file_name).exists():
#                             exist_model_file = False
#                     if exist_model_file:
#                         dynamics.load(exp_dir)
#                         print("loaded dynamics from {}".format(exp_dir))
#                         return True
#     return False
