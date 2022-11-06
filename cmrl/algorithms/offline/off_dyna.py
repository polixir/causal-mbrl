import os
from typing import Optional
from functools import partial

import emei
from omegaconf import DictConfig
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.models.util import load_offline_data

# from cmrl.models.dynamics import ConstraintBasedDynamics
from cmrl.sb3_extension.eval_callback import EvalCallback
from cmrl.models.fake_env import VecFakeEnv
from cmrl.sb3_extension.logger import configure as logger_configure
from cmrl.utils.types import InitObsFnType, RewardFnType, TermFnType
from cmrl.utils.creator import create_dynamics, create_agent


def train(
    env: emei.EmeiEnv,
    eval_env: emei.EmeiEnv,
    reward_fn: Optional[RewardFnType],
    termination_fn: Optional[TermFnType],
    get_init_obs_fn: Optional[InitObsFnType],
    cfg: DictConfig,
    work_dir: Optional[str] = None,
):
    #########################################
    # create class
    #########################################
    work_dir = work_dir or os.getcwd()
    logger = logger_configure("log", ["tensorboard", "multi_csv", "stdout"])

    # create ``cmrl`` dynamics
    dynamics = create_dynamics(cfg, env.observation_space, env.action_space, logger=logger)

    # create sb3's replay buffer for real offline data
    real_replay_buffer = ReplayBuffer(
        cfg.task.num_steps, env.observation_space, env.action_space, cfg.device, handle_timeout_termination=False
    )

    partial_fake_env = partial(
        VecFakeEnv,
        cfg.algorithm.num_envs,
        env.observation_space,
        env.action_space,
        dynamics,
        reward_fn,
        termination_fn,
        get_init_obs_fn,
        real_replay_buffer,
        penalty_coeff=cfg.task.penalty_coeff,
        logger=logger,
    )
    fake_env = partial_fake_env(
        deterministic=cfg.algorithm.deterministic, max_episode_steps=env.spec.max_episode_steps, branch_rollout=False
    )
    fake_eval_env = partial_fake_env(deterministic=True, max_episode_steps=env.spec.max_episode_steps, branch_rollout=False)

    # create sb3's agent
    agent = create_agent(cfg, fake_env, logger)

    #########################################
    # learn
    #########################################
    load_offline_data(env, real_replay_buffer, cfg.task.dataset, cfg.task.use_ratio)

    # if hasattr(env, "get_causal_graph"):
    #     oracle_causal_graph = env.get_causal_graph()
    # else:
    #     oracle_causal_graph = None
    #
    # if isinstance(dynamics, ConstraintBasedDynamics):
    #     dynamics.set_oracle_mask("transition", oracle_causal_graph.T)
    #
    # existed_trained_model = maybe_load_trained_offline_model(dynamics, cfg, obs_shape, act_shape, work_dir=work_dir)
    # if not existed_trained_model:
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

    agent.learn(total_timesteps=cfg.task.num_steps, callback=eval_callback)
