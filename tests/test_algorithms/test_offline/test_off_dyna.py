import numpy as np
import torch

from cmrl.algorithms.offline.off_dyna import train
from cmrl.util.env import make_env

from tests.constants import cfg


def test_off_dyna():
    env, term_fn, reward_fn, init_obs_fn = make_env(cfg)
    test_env, *_ = make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train(env, test_env, term_fn, reward_fn, init_obs_fn, cfg)
