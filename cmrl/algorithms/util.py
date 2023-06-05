from typing import Optional, cast
from copy import deepcopy
import pathlib

import hydra
from omegaconf import DictConfig, OmegaConf

from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.types import InitObsFnType, RewardFnType, TermFnType

from cmrl.models.dynamics import Dynamics
from cmrl.utils.config import load_hydra_cfg


def compare_dict(dict1, dict2):
    if len(list(dict1)) != len(list(dict2)):
        return False
    for key in dict1:
        if key not in dict2:
            return False
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                if not compare_dict(dict1[key], dict2[key]):
                    return False
            else:
                if dict1[key] != dict2[key]:
                    return False
    return True


def maybe_load_offline_model(
        dynamics: Dynamics,
        cfg: DictConfig,
        work_dir,
):
    work_dir = pathlib.Path(work_dir)
    if "." not in work_dir.name:  # exp by hydra's MULTIRUN mode
        task_exp_dir = work_dir.parent.parent
    else:
        task_exp_dir = work_dir.parent

    transition_cfg = OmegaConf.to_container(cfg.transition, resolve=True)

    for time_dir in task_exp_dir.glob(r"*"):
        if (time_dir / "multirun.yaml").exists():  # exp by hydra's MULTIRUN mode, multi exp in this time
            this_time_exp_dir_list = list(time_dir.glob(r"*"))
        else:  # only one exp in this time
            this_time_exp_dir_list = [time_dir]

        for exp_dir in this_time_exp_dir_list:
            if not (exp_dir / ".hydra").exists():
                continue
            exp_cfg = load_hydra_cfg(exp_dir)

            exp_transition_dir = OmegaConf.to_container(exp_cfg.transition, resolve=True)
            if (
                    cfg.seed == exp_cfg.seed
                    and cfg.task.use_ratio == exp_cfg.task.use_ratio
                    and compare_dict(exp_transition_dir, transition_cfg)
                    and (exp_dir / "transition").exists()
            ):
                dynamics.transition.load(exp_dir / "transition")
                print("loaded dynamics from {}".format(exp_dir))
                return True
    return False
