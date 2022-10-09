# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import gym
import numpy as np
import torch
from gym.core import ActType, ObsType
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

import cmrl.types
from cmrl.models.dynamics import BaseDynamics


class VecFakeEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ):
        super(VecFakeEnv, self).__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self.has_set_up = False

        self.penalty_coeff = None
        self.deterministic = None
        self.max_episode_steps = None

        self.dynamics = None
        self.reward_fn = None
        self.termination_fn = None
        self.learned_reward = None
        self.learned_termination = None
        self.get_batch_init_obs_fn = None
        self.generator = np.random.default_rng()
        self.device = None

        self._current_batch_obs = None
        self._current_batch_action = None

        self._envs_length = np.zeros(self.num_envs, dtype=int)

    def set_up(
        self,
        dynamics: BaseDynamics,
        reward_fn: Optional[cmrl.types.RewardFnType] = None,
        termination_fn: Optional[cmrl.types.TermFnType] = None,
        get_init_obs_fn: Optional[cmrl.types.InitObsFnType] = None,
        penalty_coeff: float = 0.0,
        deterministic=False,
        max_episode_steps=1000,
        logger=None,
    ):
        self.dynamics = dynamics

        self.penalty_coeff = penalty_coeff
        self.deterministic = deterministic
        self.max_episode_steps = max_episode_steps

        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        assert self.dynamics.learned_reward or reward_fn
        assert self.dynamics.learned_termination or termination_fn
        self.learned_reward = self.dynamics.learned_reward
        self.learned_termination = self.dynamics.learned_termination
        self.get_batch_init_obs_fn = get_init_obs_fn
        self.logger = logger

        self.device = dynamics.device
        self.has_set_up = True

    def step_async(self, actions: np.ndarray) -> None:
        self._current_batch_action = actions

    def step_wait(self):
        assert self.has_set_up, "fake-env has not set up"
        assert len(self._current_batch_action.shape) == 2  # batch, action_dim
        with torch.no_grad():
            batch_obs_tensor = torch.from_numpy(self._current_batch_obs).to(torch.float32).to(self.device)
            batch_action_tensor = torch.from_numpy(self._current_batch_action).to(torch.float32).to(self.device)
            dynamics_pred = self.dynamics.query(batch_obs_tensor, batch_action_tensor, return_as_np=True)

            # transition
            batch_next_obs = self.get_dynamics_predict(dynamics_pred, "transition", deterministic=self.deterministic)
            if self.learned_reward:
                batch_reward = self.get_dynamics_predict(dynamics_pred, "reward_mech", deterministic=self.deterministic)
            else:
                batch_reward = self.reward_fn(batch_next_obs, self._current_batch_obs, self._current_batch_action)
            if self.learned_termination:
                batch_terminal = self.get_dynamics_predict(dynamics_pred, "termination_mech", deterministic=self.deterministic)
            else:
                batch_terminal = self.termination_fn(batch_next_obs, self._current_batch_obs, self._current_batch_action)

            if self.penalty_coeff != 0:
                penalty = self.get_penalty(dynamics_pred["batch_next_obs"]["mean"]).reshape(batch_reward.shape)
                batch_reward -= penalty * self.penalty_coeff

                if self.logger is not None:
                    self.logger.record_mean("rollout/penalty", penalty.mean().item())

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
        if self.has_set_up:
            self._current_batch_obs = self.get_batch_init_obs_fn(self.num_envs)
            self._envs_length = np.zeros(self.num_envs, dtype=int)
            return self._current_batch_obs.copy()

    def seed(self, seed: Optional[int] = None):
        self.generator = np.random.default_rng(seed=seed)

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.num_envs)]

    def single_reset(self, idx, init_batch_obs=None):
        assert self.has_set_up, "fake-env has not set up"

        self._envs_length[idx] = 0
        if init_batch_obs is None:
            assert self.get_batch_init_obs_fn is not None
            self._current_batch_obs[idx] = self.get_batch_init_obs_fn(1)
        else:
            assert len(init_batch_obs.shape) == 2  # batch, obs_dim
            pass

    def render(self, mode="human"):
        raise NotImplementedError

    @staticmethod
    def get_penalty(ensemble_batch_next_obs):
        avg = np.mean(ensemble_batch_next_obs, axis=0)  # average predictions over models
        diffs = ensemble_batch_next_obs - avg
        dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
        penalty = np.max(dists, axis=0)  # max distances over models

        return penalty

    def get_dynamics_predict(
        self,
        origin_predict: Dict,
        mech: str,
        deterministic: bool = False,
    ):
        variable = self.dynamics.get_variable_by_mech(mech)
        ensemble_mean, ensemble_logvar = (
            origin_predict[variable]["mean"],
            origin_predict[variable]["logvar"],
        )
        batch_size = ensemble_mean.shape[1]
        random_index = getattr(self.dynamics, mech).get_random_index(batch_size, self.generator)
        if deterministic:
            pred = ensemble_mean[random_index, np.arange(batch_size)]
        else:
            ensemble_std = np.sqrt(np.exp(ensemble_logvar))
            pred = ensemble_mean[random_index, np.arange(batch_size)] + ensemble_std[
                random_index, np.arange(batch_size)
            ] * self.generator.normal(size=ensemble_mean.shape[1:]).astype(np.float32)
        return pred

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


class GymBehaviouralFakeEnv(gym.Env):
    def __init__(self, vec_fake_env: VecFakeEnv, real_env):
        assert vec_fake_env.num_envs == 1, "gym env should be single"
        self.vec_fake_env = vec_fake_env
        self.real_env = real_env

        self.current_obs = None

        self.action_space = self.vec_fake_env.action_space
        self.observation_space = self.vec_fake_env.observation_space

    def step(self, action):
        batch_next_obs, batch_reward, batch_terminal, _ = self.vec_fake_env.step(np.array([action], dtype=np.float32))
        self.current_obs = batch_next_obs[0]
        return batch_next_obs[0], batch_reward[0], batch_terminal[0], {}

    def render(self, mode="human"):
        assert mode == "human"
        self.real_env.freeze()
        self.real_env.set_state_by_obs(self.current_obs)
        self.real_env.render()
        self.real_env.unfreeze()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        batch_next_obs = self.vec_fake_env.reset()
        self.current_obs = batch_next_obs[0]
        return batch_next_obs[0]
