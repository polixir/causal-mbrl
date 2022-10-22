import os
from typing import Optional, cast

import emei
import hydra.utils
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.agent import complete_agent_cfg
from cmrl.algorithms.util import maybe_load_trained_offline_model, setup_fake_env, load_offline_data
from cmrl.models.dynamics import ConstraintBasedDynamics
from cmrl.sb3_extension.eval_callback import EvalCallback
from cmrl.sb3_extension.logger import configure as logger_configure
from cmrl.types import InitObsFnType, RewardFnType, TermFnType
from cmrl.util.creator import create_dynamics


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

    # create initial dataset and add it to replay buffer
    dynamics = create_dynamics(cfg.dynamics, obs_shape, act_shape, logger=logger)
    real_replay_buffer = ReplayBuffer(
        cfg.task.num_steps, env.observation_space, env.action_space, cfg.device, handle_timeout_termination=False
    )
    load_offline_data(cfg, env, real_replay_buffer)

    fake_eval_env = setup_fake_env(
        cfg=cfg,
        agent=agent,
        dynamics=dynamics,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        get_init_obs_fn=get_init_obs_fn,
        logger=logger,
        max_episode_steps=env.spec.max_episode_steps,
        penalty_coeff=cfg.algorithm.penalty_coeff,
    )

    if hasattr(env, "get_causal_graph"):
        oracle_causal_graph = env.get_causal_graph()
    else:
        oracle_causal_graph = None

    if isinstance(dynamics, ConstraintBasedDynamics):
        dynamics.set_oracle_mask("transition", oracle_causal_graph.T)

    existed_trained_model = maybe_load_trained_offline_model(dynamics, cfg, obs_shape, act_shape, work_dir=work_dir)
    if not existed_trained_model:
        dynamics.learn(real_replay_buffer, **cfg.dynamics, work_dir=work_dir)

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
