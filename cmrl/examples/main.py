import hydra
import numpy as np
import omegaconf
import torch

import cmrl.algorithms.offline.mopo as mopo
import cmrl.algorithms.online.mbpo as mbpo

from cmrl.util.env import make_env


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = make_env(cfg)
    test_env, *_ = make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.algorithm.name == "mopo":
        test_env, *_ = make_env(cfg)
        return mopo.train(env, test_env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = make_env(cfg)
        return mbpo.train(env, test_env, term_fn, reward_fn, cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    run()
