import os
from typing import Optional, Sequence, cast

import gym
import emei
import numpy as np

import cmrl.constants
import cmrl.models
import cmrl.agent
import cmrl.types
from cmrl.agent.sac_wrapper import SACAgent
from cmrl.util.video import VideoRecorder


def rollout_model_and_populate_sac_buffer(
        query_env: emei.EmeiEnv,
        fake_env: cmrl.models.FakeEnv,
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
        pred_next_obs, pred_rewards, pred_dones = fake_env.step(
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
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()

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
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
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
