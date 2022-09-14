import hydra
import numpy as np
import omegaconf
import torch

import cmrl.algorithms.offline.mopo as mopo
import cmrl.algorithms.online.mbpo as mbpo
import cmrl.algorithms.offline.off_dyna as off_dyna

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
    elif cfg.algorithm.name == "off_dyna":
        test_env, *_ = make_env(cfg)
        get_init_obs_fn = env.get_batch_init_obs
        return off_dyna.train(env, test_env, term_fn, reward_fn, get_init_obs_fn, cfg)
    elif cfg.algorithm.name == "mbpo":
        test_env, *_ = make_env(cfg)
        return mbpo.train(env, test_env, term_fn, reward_fn, cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    run()
