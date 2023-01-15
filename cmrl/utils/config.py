import pathlib
from typing import Dict, Union, Optional
from collections import defaultdict

import omegaconf
from omegaconf import DictConfig
import pandas as pd
import numpy as np

PACKAGE_PATH = pathlib.Path(__file__).parent.parent.parent


def load_hydra_cfg(results_dir: Union[str, pathlib.Path]) -> omegaconf.DictConfig:
    """Loads a Hydra configuration from the given directory path.

    Tries to load the configuration from "results_dir/.hydra/config.yaml".

    Args:
        results_dir (str or pathlib.Path): the path to the directory containing the config.

    Returns:
        (omegaconf.DictConfig): the loaded configuration.

    """
    results_dir = pathlib.Path(results_dir)
    cfg_file = results_dir / ".hydra" / "config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_file)
    if not isinstance(cfg, omegaconf.DictConfig):
        raise RuntimeError("Configuration format not a omegaconf.DictConf")
    return cfg


def exp_collect(cfg_extractor,
                csv_extractor,
                env_name="ContinuousCartPoleSwingUp-v0",
                exp_name="default",
                exp_path=None):
    data = defaultdict(list)

    if exp_path is None:
        exp_path = PACKAGE_PATH / "exp"
    exp_dir = exp_path / exp_name
    env_dir = exp_dir / env_name

    for params_dir in env_dir.glob("*"):
        for dataset_dir in params_dir.glob("*"):
            for time_dir in dataset_dir.glob("*"):
                if not (time_dir / ".hydra").exists():  # exp by hydra's MULTIRUN mode, multi exp in this time
                    time_dir_list = list(time_dir.glob("*"))
                else:  # only one exp in this time
                    time_dir_list = [time_dir]

                for single_dir in time_dir_list:
                    if single_dir.name == "multirun.yaml":
                        continue

                    cfg = load_hydra_cfg(single_dir)

                    key = cfg_extractor(cfg, params_dir.name, dataset_dir.name, time_dir.name)
                    if not key:
                        continue

                    data[key] = csv_extractor(single_dir / "log")
    return data
