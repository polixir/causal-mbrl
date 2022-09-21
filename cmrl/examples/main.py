import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

import cmrl.algorithms.offline.mopo as mopo
import cmrl.algorithms.online.mbpo as mbpo
import cmrl.algorithms.offline.off_dyna as off_dyna
from cmrl.util.env import make_env


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run(cfg: DictConfig):
    if cfg.wandb:
        wandb.init(
            project='causal-mbrl',
            group=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True
        )

    env, term_fn, reward_fn, init_obs_fn = make_env(cfg)
    test_env, *_ = make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.algorithm.name == "mopo":
        test_env, *_ = make_env(cfg)
        return mopo.train(env, test_env, term_fn, reward_fn, cfg)
    elif cfg.algorithm.name == "off_dyna":
        test_env, *_ = make_env(cfg)
        return off_dyna.train(env, test_env, term_fn, reward_fn, init_obs_fn, cfg)
    elif cfg.algorithm.name == "mbpo":
        test_env, *_ = make_env(cfg)
        return mbpo.train(env, test_env, term_fn, reward_fn, cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    run()
