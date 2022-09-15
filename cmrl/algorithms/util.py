import os
import pathlib
from typing import Optional, Sequence, cast, Dict, Callable, List, Tuple

import gym
import emei
import numpy as np

import cmrl.constants
import cmrl.models
import cmrl.agent
import cmrl.types
from omegaconf import DictConfig
from cmrl.agent.sac_wrapper import SACAgent
from cmrl.util.video import VideoRecorder
from cmrl.util.config import load_hydra_cfg, get_complete_dynamics_cfg
from cmrl.util.replay_buffer import ReplayBuffer
from cmrl.models.dynamics import BaseDynamics


def rollout_model_and_populate_sac_buffer(
        query_env: emei.EmeiEnv,
        fake_env: cmrl.models.VecFakeEnv,
        replay_buffer: cmrl.util.ReplayBuffer,
        agent: SACAgent,
        sac_buffer: cmrl.util.ReplayBuffer,
        sac_samples_action: bool,
        rollout_horizon: int,
        batch_size: int,
        logger,
        epoch: int,
        query_ratio: float = 5e-3
):
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(cmrl.types.InteractionBatch, batch).as_tuple()
    fake_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs

    query_index = np.random.choice(batch_size, int(query_ratio * batch_size))
    gt_obs = obs[query_index]
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_reward, pred_terminal = fake_env.step(
            action, deterministic=False
        )
        # query ground truth
        # gt_next_obs, gt_rewards, gt_dones, gt_info = query_env.query(gt_obs, action[query_index])
        # next_obs_loss = ((gt_next_obs - pred_next_obs[query_index]) ** 2).mean(axis=0)
        # rewards_loss = ((gt_rewards - pred_rewards[query_index]) ** 2).mean()
        # log_data = [("epoch", epoch), ("rollout", i)] + [("obs{}".format(o), l) for o, l in
        #                                                  enumerate(next_obs_loss)] + [("reward", rewards_loss)]
        # logger.log_data(
        #     "model_eval",
        #     dict(log_data)
        # )

        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_reward[~accum_dones, 0],
            pred_terminal[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_terminal.squeeze()

        if accum_dones.all():
            print("stop by env, longest rollout:", i)
            break
        if obs.max() > 1000:
            print("stop by wrong obs, longest rollout:", i)
            break


def evaluate(
        env: gym.Env,
        agent: SACAgent,
        num_episodes: int,
        video_recorder: VideoRecorder,
) -> [float, float]:
    episodes_reward = []
    episodes_length = []
    for episode in range(num_episodes):
        obs = env.reset()
        video_recorder.init(enabled=(episode == 0))
        terminal = truncated = False
        episode_reward = 0
        episode_length = 0
        while not (terminal or truncated):
            action = agent.act(obs)
            obs, reward, terminal, truncated, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
            episode_length += 1
        episodes_reward.append(episode_reward)
        episodes_length.append(episode_length)
    return np.array(episodes_reward), np.array(episodes_length)


def maybe_replace_sac_buffer(
        sac_buffer: Optional[cmrl.util.ReplayBuffer],
        obs_shape: Sequence[int],
        act_shape: Sequence[int],
        new_capacity: int,
        seed: int,
) -> cmrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = cmrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        obs, action, next_obs, reward, done = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, done)
        return new_buffer
    return sac_buffer


def truncated_linear(
        min_x: float, max_x: float, min_y: float, max_y: float, x: float
) -> float:
    """Truncated linear function.

    Implements the following function:
        f1(x) = min_y + (x - min_x) / (max_x - min_x) * (max_y - min_y)
        f(x) = min(max_y, max(min_y, f1(x)))

    If max_x - min_x < 1e-10, then it behaves as the constant f(x) = max_y
    """
    if max_x - min_x < 1e-10:
        return max_y
    if x <= min_x:
        y: float = min_y
    else:
        dx = (x - min_x) / (max_x - min_x)
        dx = min(dx, 1.0)
        y = dx * (max_y - min_y) + min_y
    return y


def is_same_dict(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            # print(0, key)
            return False
        else:
            if isinstance(dict1[key], DictConfig) and isinstance(dict2[key], DictConfig):
                if not is_same_dict(dict1[key], dict2[key]):
                    # print(1, key, dict1[key], dict2[key])
                    return False
            else:
                if dict1[key] != dict2[key]:
                    # print(2, key, dict1[key], dict2[key])
                    return False
    return True


def maybe_load_trained_offline_model(dynamics: BaseDynamics,
                                     cfg,
                                     obs_shape,
                                     act_shape,
                                     work_dir):
    work_dir = pathlib.Path(work_dir)
    if "." not in work_dir.name:  # exp by hydra's MULTIRUN mode
        task_exp_dir = work_dir.parent.parent.parent
    else:
        task_exp_dir = work_dir.parent.parent
    dynamics_cfg = cfg.dynamics

    for date_dir in task_exp_dir.glob(r"*"):
        for time_dir in date_dir.glob(r"*"):
            if (time_dir / "multirun.yaml").exists():  # exp by hydra's MULTIRUN mode, multi exp in this time
                this_time_exp_dir_list = list(time_dir.glob(r"*"))
            else:  # only one exp in this time
                this_time_exp_dir_list = [time_dir]

            for exp_dir in this_time_exp_dir_list:
                if not (exp_dir / ".hydra").exists():
                    continue
                exp_cfg = load_hydra_cfg(exp_dir)
                exp_dynamics_cfg = get_complete_dynamics_cfg(exp_cfg.dynamics, obs_shape, act_shape)

                if exp_cfg.seed == cfg.seed and is_same_dict(dynamics_cfg, exp_dynamics_cfg):
                    exist_model_file = True
                    for mech in dynamics.learn_mech:
                        mech_file_name = getattr(dynamics, mech).model_file_name
                        if not (exp_dir / mech_file_name).exists():
                            exist_model_file = False
                    if exist_model_file:
                        dynamics.load(exp_dir)
                        print("loaded dynamics from {}".format(exp_dir))
                        return True
    return False


def rollout_agent_trajectories(
        env: gym.Env,
        steps_or_trials_to_collect: int,
        agent: cmrl.agent.Agent,
        agent_kwargs: Dict,
        trial_length: Optional[int] = None,
        callback: Optional[Callable] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        collect_full_trajectories: bool = False,
        agent_uses_low_dim_obs: bool = False,
) -> List[float]:
    """Rollout agent trajectories in the given environment.

    Rollouts trajectories in the environment using actions produced by the given agent.
    Optionally, it stores the saved data into a replay buffer.

    Args:
        env (gym.Env): the environment to step.
        steps_or_trials_to_collect (int): how many steps of the environment to collect. If
            ``collect_trajectories=True``, it indicates the number of trials instead.
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        trial_length (int, optional): a maximum length for trials (env will be reset regularly
            after this many number of steps). Defaults to ``None``, in which case trials
            will end when the environment returns ``done=True``.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, done)`.
        replay_buffer (:class:`cmrl.util.ReplayBuffer`, optional):
            a replay buffer to store data to use for training.
        collect_full_trajectories (bool): if ``True``, indicates that replay buffers should
            collect full trajectories. This only affects the split between training and
            validation buffers. If ``collect_trajectories=True``, the split is done over
            trials (full trials in each dataset); otherwise, it's done across steps.
        agent_uses_low_dim_obs (bool): only valid if env is of type
            :class:`cmrl.env.MujocoGymPixelWrapper` and replay_buffer is not ``None``.
            If ``True``, instead of passing the obs
            produced by env.reset/step to the agent, it will pass
            obs = env.get_last_low_dim_obs(). This is useful for rolling out an agent
            trained with low dimensional obs, but collect pixel obs in the replay buffer.

    Returns:
        (list(float)): Total rewards obtained at each complete trial.
    """
    if (
            replay_buffer is not None
            and replay_buffer.stores_trajectories
            and not collect_full_trajectories
    ):
        # Might be better as a warning but it's possible that users will miss it.
        raise RuntimeError(
            "Replay buffer is tracking trajectory information but "
            "collect_trajectories is set to False, which will result in "
            "corrupted trajectory data."
        )

    step = 0
    trial = 0
    total_rewards: List[float] = []
    while True:
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        while not done:
            if replay_buffer is not None:
                next_obs, reward, done, info = step_env_and_add_to_buffer(
                    env,
                    obs,
                    agent,
                    agent_kwargs,
                    replay_buffer,
                    callback=callback,
                    agent_uses_low_dim_obs=agent_uses_low_dim_obs,
                )
            else:
                if agent_uses_low_dim_obs:
                    raise RuntimeError(
                        "Option agent_uses_low_dim_obs is only valid if a "
                        "replay buffer is given."
                    )
                action = agent.act(obs, **agent_kwargs)
                next_obs, reward, done, info = env.step(action)
                if callback:
                    callback((obs, action, next_obs, reward, done))
            obs = next_obs
            total_reward += reward
            step += 1
            if not collect_full_trajectories and step == steps_or_trials_to_collect:
                total_rewards.append(total_reward)
                return total_rewards
            if trial_length and step % trial_length == 0:
                if collect_full_trajectories and not done and replay_buffer is not None:
                    replay_buffer.close_trajectory()
                break
        trial += 1
        total_rewards.append(total_reward)
        if collect_full_trajectories and trial == steps_or_trials_to_collect:
            break
    return total_rewards


def step_env_and_add_to_buffer(
        env: gym.Env,
        obs: np.ndarray,
        agent: cmrl.agent.Agent,
        agent_kwargs: Dict,
        replay_buffer: ReplayBuffer,
        callback: Optional[Callable] = None,
        agent_uses_low_dim_obs: bool = False,
) -> Tuple[np.ndarray, float, bool, Dict]:
    """Steps the environment with an agent's action and populates the replay buffer.

    Args:
        env (gym.Env): the environment to step.
        obs (np.ndarray): the latest observation returned by the environment (used to obtain
            an action from the agent).
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer
            containing stored data.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, done)`.
        agent_uses_low_dim_obs (bool): only valid if env is of type
            :class:`cmrl.env.MujocoGymPixelWrapper`. If ``True``, instead of passing the obs
            produced by env.reset/step to the agent, it will pass
            obs = env.get_last_low_dim_obs(). This is useful for rolling out an agent
            trained with low dimensional obs, but collect pixel obs in the replay buffer.

    Returns:
        (tuple): next observation, reward, done and meta-info, respectively, as generated by
        `env.step(agent.act(obs))`.
    """

    if agent_uses_low_dim_obs and not hasattr(env, "get_last_low_dim_obs"):
        raise RuntimeError(
            "Option agent_uses_low_dim_obs is only compatible with "
            "env of type cmrl.env.MujocoGymPixelWrapper."
        )
    if agent_uses_low_dim_obs:
        agent_obs = getattr(env, "get_last_low_dim_obs")()
    else:
        agent_obs = obs
    action = agent.act(agent_obs, **agent_kwargs)
    next_obs, reward, done, info = env.step(action)
    replay_buffer.add(obs, action, next_obs, reward, done)
    if callback:
        callback((obs, action, next_obs, reward, done))
    return next_obs, reward, done, info
