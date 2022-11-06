import pathlib
from typing import Tuple, Union

import omegaconf


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


def get_complete_dynamics_cfg(
    dynamics_cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
):
    transition_cfg = dynamics_cfg.transition
    transition_cfg.obs_size = obs_shape[0]
    transition_cfg.action_size = act_shape[0]

    reward_cfg = dynamics_cfg.reward_mech
    reward_cfg.obs_size = obs_shape[0]
    reward_cfg.action_size = act_shape[0]

    termination_cfg = dynamics_cfg.termination_mech
    termination_cfg.obs_size = obs_shape[0]
    termination_cfg.action_size = act_shape[0]
    return dynamics_cfg
