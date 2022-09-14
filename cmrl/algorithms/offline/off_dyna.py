import os
from typing import Optional, Sequence, cast

import gym
from gym.wrappers import TimeLimit
import emei
import hydra.utils
import numpy as np
import omegaconf
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback

import cmrl.constants
import cmrl.agent
import cmrl.models
import cmrl.models.dynamics
import cmrl.third_party.pytorch_sac as pytorch_sac
import cmrl.types
import cmrl.util
import cmrl.util.creator as creator
from cmrl.agent.sac_wrapper import SACAgent
from cmrl.util.video import VideoRecorder
from cmrl.algorithms.util import evaluate, rollout_model_and_populate_sac_buffer, maybe_replace_sac_buffer, \
    truncated_linear, maybe_load_trained_offline_model

MBPO_LOG_FORMAT = cmrl.constants.RESULTS_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]
MODEL_EVAL_LOG_FORMAT = [
    ("epoch", "E", "int"),
    ("rollout", "RO", "int"),
]


def train(
        env: emei.EmeiEnv,
        test_env: emei.EmeiEnv,
        termination_fn: Optional[cmrl.types.TermFnType],
        reward_fn: Optional[cmrl.types.RewardFnType],
        get_init_obs_fn,
        cfg: omegaconf.DictConfig,
        silent: bool = False,
        work_dir: Optional[str] = None,
) -> np.float32:
    """Train agent by MOPO algorithm.

    Args:
        env: interaction environment
        test_env: test environment, only used to evaluation
        termination_fn: termination function given as priori, `None` if it needs to be learned by nn
        reward_fn: reward function given as priori, `None` if it needs to be learned by nn
        cfg: all config
        silent: no logging
        work_dir:

    Returns: the best evaluation reward
    """
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cmrl.agent.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = cast(BaseAlgorithm, hydra.utils.instantiate(cfg.algorithm.agent))

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = cmrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        cmrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    logger.register_group(
        "model_eval",
        [("obs{}".format(o), "O{}".format(o), "float") for o in range(obs_shape[0])] + [
            ("reward", "R", "float")] + MODEL_EVAL_LOG_FORMAT,
        color="green",
        dump_frequency=1,
        disable_console_dump=True
    )
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)
    numpy_generator = np.random.default_rng(seed=cfg.seed)

    # -------------- Create initial dataset --------------
    dynamics = creator.create_dynamics(cfg.dynamics, obs_shape, act_shape, logger=logger)
    replay_buffer = creator.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        numpy_generator=numpy_generator,
    )
    # load replay buffer data
    if hasattr(env, "get_dataset"):
        params, dataset_type = cfg.task.env.split("___")[-2:]
        data_dict = env.get_dataset("{}-{}".format(params, dataset_type))
        replay_buffer.add_batch(data_dict["observations"],
                                data_dict["actions"],
                                data_dict["next_observations"],
                                data_dict["rewards"],
                                data_dict["terminals"].astype(bool) | data_dict["timeouts"].astype(bool))
    else:
        raise NotImplementedError

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
            cfg.task.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.task.epoch_length / cfg.task.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    fake_env = agent.env.envs[0].env
    fake_env.complete(dynamics,
                               reward_fn,
                               termination_fn,
                               get_init_obs_fn,
                               generator=numpy_generator,
                               penalty_coeff=cfg.algorithm.penalty_coeff)
    agent.env.envs[0] = TimeLimit(fake_env, max_episode_steps=env.spec.max_episode_steps, new_step_api=False)

    if hasattr(env, "causal_graph"):
        oracle_causal_graph = env.causal_graph
    else:
        oracle_causal_graph = None

    if isinstance(dynamics, cmrl.models.dynamics.ConstraintBasedDynamics):
        dynamics.set_oracle_mask("transition", oracle_causal_graph[:-1])

    existed_trained_model = maybe_load_trained_offline_model(dynamics, cfg, obs_shape, act_shape,
                                                             work_dir=work_dir)
    if not existed_trained_model:
        dynamics.learn(replay_buffer,
                       **cfg.dynamics,
                       work_dir=work_dir)

    eval_callback = EvalCallback(test_env, best_model_save_path="./",
                                 log_path="./", eval_freq=1000,
                                 deterministic=True, render=False)

    agent.learn(total_timesteps=int(1e6), callback=eval_callback, tb_log_name="./tb")
