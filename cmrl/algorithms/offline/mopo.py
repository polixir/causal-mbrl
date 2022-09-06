import os
from typing import Optional, Sequence, cast

import emei
import hydra.utils
import numpy as np
import omegaconf
import torch

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
    truncated_linear, maybe_load_trained_model

MBPO_LOG_FORMAT = cmrl.constants.EVAL_LOG_FORMAT + [
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
    agent = SACAgent(
        cast(pytorch_sac.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

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
    fake_env = cmrl.models.FakeEnv(env,
                                   dynamics,
                                   reward_fn,
                                   termination_fn,
                                   generator=numpy_generator,
                                   penalty_coeff=cfg.algorithm.penalty_coeff)
    if hasattr(env, "causal_graph"):
        oracle_causal_graph = env.causal_graph
    else:
        oracle_causal_graph = None

    if isinstance(dynamics, cmrl.models.dynamics.ConstraintBasedDynamics):
        dynamics.set_oracle_mask("transition", oracle_causal_graph[:-1])

    existed_trained_model = maybe_load_trained_model(dynamics, cfg, obs_shape, act_shape,
                                                     work_dir=work_dir)
    if not existed_trained_model:
        dynamics.learn(replay_buffer,
                       **cfg.dynamics,
                       work_dir=work_dir)

    best_eval_reward = -np.inf
    sac_buffer = None

    for epoch in range(cfg.task.num_steps // cfg.task.epoch_length):
        rollout_length = int(
            truncated_linear(
                *(cfg.task.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.task.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        for steps_epoch in range(cfg.task.epoch_length):
            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.task.freq_train_model == 0:
                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                rollout_model_and_populate_sac_buffer(
                    test_env,
                    fake_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                    logger,
                    epoch
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.task.num_sac_updates_per_step):
                use_real_data = numpy_generator.random() < cfg.algorithm.real_data_ratio
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.task.sac_updates_every_steps != 0 or len(
                        which_buffer
                ) < cfg.task.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.task.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.task.test_freq == 0:
                rewards, lengths = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                logger.log_data(
                    cmrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "reward_mean": rewards.mean(),
                        "reward_std": rewards.std(),
                        "length_mean": lengths.mean(),
                        "length_std": lengths.std(),
                        "rollout_length": rollout_length,
                    },
                )
                agent.sac_agent.save_checkpoint(ckpt_path=os.path.join(work_dir, "sac_final.pth"))
                if rewards.mean() > best_eval_reward:
                    video_recorder.save(f"{epoch}.mp4")
                    best_eval_reward = rewards.mean()
                    agent.sac_agent.save_checkpoint(ckpt_path=os.path.join(work_dir, "sac_best.pth"))

            env_steps += 1
    return np.float32(best_eval_reward)
