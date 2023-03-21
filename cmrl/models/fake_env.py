from typing import Any, Dict, List, Optional, Type

import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.types import RewardFnType, TermFnType, InitObsFnType
from cmrl.models.dynamics import Dynamics


def get_penalty(ensemble_batch_next_obs):
    avg = np.mean(ensemble_batch_next_obs, axis=0)  # average predictions over models
    diffs = ensemble_batch_next_obs - avg
    dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
    penalty = np.max(dists, axis=0)  # max distances over models
    return penalty


class VecFakeEnv(VecEnv):
    def __init__(
            self,
            # for need of sb3's agent
            num_envs: int,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            # for dynamics
            dynamics: Dynamics,
            reward_fn: Optional[RewardFnType] = None,
            termination_fn: Optional[TermFnType] = None,
            get_init_obs_fn: Optional[InitObsFnType] = None,
            real_replay_buffer: Optional[ReplayBuffer] = None,
            # for offline
            penalty_coeff: float = 0.0,
            # for behaviour
            deterministic: bool = False,
            max_episode_steps: int = 1000,
            branch_rollout: bool = False,
            # others
            logger: Optional[Logger] = None,
            **kwargs,
    ):
        super(VecFakeEnv, self).__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        assert self.dynamics.learn_reward or reward_fn, "you must learn a reward-mech or give one"
        assert self.dynamics.learn_termination or termination_fn, "you must learn a termination-mech or give one"
        self.learn_reward = self.dynamics.learn_reward
        self.learn_termination = self.dynamics.learn_termination
        self.get_init_obs_fn = get_init_obs_fn
        self.replay_buffer = real_replay_buffer

        self.penalty_coeff = penalty_coeff
        self.deterministic = deterministic
        self.max_episode_steps = max_episode_steps
        self.branch_rollout = branch_rollout
        if self.branch_rollout:
            assert self.replay_buffer, "you must provide a replay buffer if using branch-rollout"
        else:
            assert self.get_init_obs_fn, "you must provide a get-init-obs function if using fully-virtual"

        self.logger = logger

        self.device = dynamics.device

        self._current_batch_obs = None
        self._current_batch_action = None
        self._envs_length = np.zeros(self.num_envs, dtype=int)

    def step_async(self, actions: np.ndarray) -> None:
        assert len(actions.shape) == 2  # batch, action_dim
        self._current_batch_action = actions

    def step_wait(self):
        batch_next_obs, batch_reward, batch_terminal, info = self.dynamics.step(
            self._current_batch_obs, self._current_batch_action
        )

        if not self.learn_reward:
            batch_reward = self.reward_fn(batch_next_obs, self._current_batch_obs, self._current_batch_action)
        if not self.learn_termination:
            batch_terminal = self.termination_fn(batch_next_obs, self._current_batch_obs, self._current_batch_action)

        if self.penalty_coeff != 0:
            penalty = get_penalty(info["origin-next_obs"]).reshape(batch_reward.shape) * self.penalty_coeff
            batch_reward -= penalty

            if self.logger is not None:
                self.logger.record_mean("rollout/penalty", penalty.mean().item())

        assert not np.isnan(batch_next_obs).any(), "next obs of fake env should not be nan."
        assert not np.isnan(batch_reward).any(), "reward of fake env should not be nan."
        assert not np.isnan(batch_terminal).any(), "terminal of fake env should not be nan."

        self._current_batch_obs = batch_next_obs.copy()
        batch_reward = batch_reward.reshape(self.num_envs)
        batch_terminal = batch_terminal.reshape(self.num_envs)
        infos = [{} for _ in range(self.num_envs)]

        self._envs_length += 1
        batch_truncate = self._envs_length >= self.max_episode_steps
        batch_done = np.logical_or(batch_terminal, batch_truncate)
        for idx in range(self.num_envs):
            infos[idx]["TimeLimit.truncated"] = batch_truncate[idx]
            if batch_done[idx]:
                infos[idx]["terminal_observation"] = batch_next_obs[idx].copy()
                self.single_reset(idx)

        return (
            self._current_batch_obs.copy(),
            batch_reward.copy(),
            batch_done.copy(),
            infos,
        )

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        if self.branch_rollout:
            upper_bound = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos
            batch_inds = np.random.randint(0, upper_bound, size=self.num_envs)
            self._current_batch_obs = self.replay_buffer.observations[batch_inds, 0]
        else:
            self._current_batch_obs = self.get_init_obs_fn(self.num_envs)
        self._envs_length = np.zeros(self.num_envs, dtype=int)

        if return_info:
            return self._current_batch_obs.copy(), {}
        else:
            return self._current_batch_obs.copy()

    def seed(self, seed: Optional[int] = None):
        self.generator = np.random.default_rng(seed=seed)

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.num_envs)]

    def single_reset(self, idx):
        self._envs_length[idx] = 0
        if self.branch_rollout:
            upper_bound = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos
            batch_idxs = np.random.randint(0, upper_bound)
            self._current_batch_obs[idx] = self.replay_buffer.observations[batch_idxs, 0]
        else:
            assert self.get_init_obs_fn is not None
            self._current_batch_obs[idx] = self.get_init_obs_fn(1)

    def render(self, mode="human"):
        raise NotImplementedError

    def env_method(
            self,
            method_name: str,
            *method_args,
            indices: VecEnvIndices = None,
            **method_kwargs,
    ) -> List[Any]:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass
