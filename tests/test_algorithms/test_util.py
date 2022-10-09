from cmrl.algorithms.util import load_offline_data

import os
from copy import deepcopy
from typing import Optional, cast

import emei
import hydra.utils
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader, Dataset

from cmrl.agent import complete_agent_cfg
from cmrl.algorithms.util import maybe_load_trained_offline_model, setup_fake_env
from cmrl.models.dynamics import ConstraintBasedDynamics
from cmrl.models.fake_env import VecFakeEnv
from cmrl.sb3_extension.eval_callback import EvalCallback
from cmrl.sb3_extension.logger import configure as logger_configure
from cmrl.types import InitObsFnType, RewardFnType, TermFnType
from cmrl.util.env import make_env

from tests.utils import cfg


def test_load_offline_data():
    env, term_fn, reward_fn, init_obs_fn = make_env(cfg)
    replay_buffer = ReplayBuffer(
        cfg.task.num_steps, env.observation_space, env.action_space, cfg.device, handle_timeout_termination=False
    )

    load_offline_data(cfg, env, replay_buffer)
