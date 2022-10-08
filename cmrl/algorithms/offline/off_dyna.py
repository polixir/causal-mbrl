import os
from copy import deepcopy
from typing import Optional, cast

import emei
import hydra.utils
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from torch.utils.data import DataLoader, Dataset

from cmrl.agent import complete_agent_cfg
from cmrl.algorithms.util import maybe_load_trained_offline_model
from cmrl.models.dynamics import ConstraintBasedDynamics
from cmrl.models.fake_env import VecFakeEnv
from cmrl.sb3_extension.eval_callback import EvalCallback
from cmrl.sb3_extension.logger import configure as logger_configure
from cmrl.types import InitObsFnType, RewardFnType, TermFnType
from cmrl.util.creator import create_dynamics, create_replay_buffer


def train(
    env: emei.EmeiEnv,
    eval_env: emei.EmeiEnv,
    termination_fn: Optional[TermFnType],
    reward_fn: Optional[RewardFnType],
    get_init_obs_fn: Optional[InitObsFnType],
    cfg: DictConfig,
    work_dir: Optional[str] = None,
):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # build model-free agent, which is a stable-baselines3's agent
    complete_agent_cfg(env, cfg.algorithm.agent)
    agent = cast(BaseAlgorithm, hydra.utils.instantiate(cfg.algorithm.agent))

    work_dir = work_dir or os.getcwd()
    logger = logger_configure("log", ["tensorboard", "multi_csv", "stdout"])

    numpy_generator = np.random.default_rng(seed=cfg.seed)

    # create initial dataset
    dynamics = create_dynamics(cfg.dynamics, obs_shape, act_shape, logger=logger)
    replay_buffer = create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        numpy_generator=numpy_generator,
    )
    # load replay buffer data
    if hasattr(env, "get_dataset"):
        params, dataset_type = cfg.task.env.split("___")[-2:]
        data_dict = env.get_dataset("{}-{}".format(params, dataset_type))
        all_data_num = len(data_dict["observations"])
        sample_data_num = int(cfg.task.use_ratio * all_data_num)
        sample_idx = numpy_generator.permutation(all_data_num)[:sample_data_num]
        replay_buffer.add_batch(
            data_dict["observations"][sample_idx],
            data_dict["actions"][sample_idx],
            data_dict["next_observations"][sample_idx],
            data_dict["rewards"][sample_idx],
            data_dict["terminals"][sample_idx].astype(bool)
            | data_dict["timeouts"][sample_idx].astype(bool),
        )
    else:
        raise NotImplementedError

    if cfg.dynamics.name == "plain_dynamics":
        penalty_coeff = cfg.algorithm.penalty_coeff
    elif cfg.dynamics.name == "constraint_based_dynamics":
        penalty_coeff = cfg.algorithm.penalty_coeff / 3
    else:
        raise NotImplementedError

    fake_env = cast(VecFakeEnv, agent.env)
    fake_env.set_up(
        dynamics,
        reward_fn,
        termination_fn,
        get_init_obs_fn,
        logger=logger,
        max_episode_steps=env.spec.max_episode_steps,
        penalty_coeff=penalty_coeff,
    )
    agent.env = VecMonitor(fake_env)

    eval_env_cfg = deepcopy(cfg.algorithm.agent.env)
    eval_env_cfg.num_envs = cfg.task.n_eval_episodes
    fake_eval_env = cast(VecFakeEnv, hydra.utils.instantiate(eval_env_cfg))
    fake_eval_env.set_up(
        dynamics,
        reward_fn,
        termination_fn,
        get_init_obs_fn,
        max_episode_steps=env.spec.max_episode_steps,
        penalty_coeff=penalty_coeff,
    )
    fake_eval_env.seed(seed=cfg.seed)

    if hasattr(env, "get_causal_graph"):
        oracle_causal_graph = env.get_causal_graph()
    else:
        oracle_causal_graph = None

    if isinstance(dynamics, ConstraintBasedDynamics):
        dynamics.set_oracle_mask("transition", oracle_causal_graph.T)

    existed_trained_model = maybe_load_trained_offline_model(
        dynamics, cfg, obs_shape, act_shape, work_dir=work_dir
    )
    if not existed_trained_model:
        dynamics.learn(replay_buffer, **cfg.dynamics, work_dir=work_dir)

    eval_callback = EvalCallback(
        eval_env,
        fake_eval_env,
        n_eval_episodes=cfg.task.n_eval_episodes,
        best_model_save_path="./",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    agent.set_logger(logger)
    agent.learn(total_timesteps=cfg.task.num_steps, callback=eval_callback)
