import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from cmrl.algorithms import mopo, mbpo, off_dyna, on_dyna
from cmrl.util.env import make_env


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run(cfg: DictConfig):
    if cfg.wandb:
        wandb.init(
            project="causal-mbrl",
            group=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
        )

    env, term_fn, reward_fn, init_obs_fn = make_env(cfg)
    test_env, *_ = make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.algorithm.name == "on_dyna":
        test_env, *_ = make_env(cfg)
        return on_dyna.train(env, test_env, term_fn, reward_fn, init_obs_fn, cfg)
    elif cfg.algorithm.name == "mopo":
        test_env, *_ = make_env(cfg)
        return mopo.train(env, test_env, term_fn, reward_fn, init_obs_fn, cfg)
    elif cfg.algorithm.name == "off_dyna":
        test_env, *_ = make_env(cfg)
        return off_dyna.train(env, test_env, term_fn, reward_fn, init_obs_fn, cfg)
    elif cfg.algorithm.name == "mbpo":
        test_env, *_ = make_env(cfg)
        return mbpo.train(env, test_env, term_fn, reward_fn, init_obs_fn, cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    run()
