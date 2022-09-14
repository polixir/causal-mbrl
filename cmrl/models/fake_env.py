# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple, Union

import gym
from gym.core import ObsType, ActType
import numpy as np
import torch

import cmrl.types
from cmrl.models.dynamics import BaseDynamics


class LazyFakeEnv(gym.Env):
    def __init__(self,
                 observation_space,
                 action_space,
                 ):
        self.completed = False

        self.penalty_coeff = None

        self.dynamics = None
        self.reward_fn = None
        self.termination_fn = None
        self.learned_reward = None
        self.learned_termination = None
        self.get_batch_init_obs_fn = None
        self.generator = None
        self.device = None
        self.observation_space = observation_space
        self.action_space = action_space

        self._current_batch_obs = None

    def complete(self,
                 dynamics: BaseDynamics,
                 reward_fn: Optional[cmrl.types.RewardFnType] = None,
                 termination_fn: Optional[cmrl.types.TermFnType] = None,
                 get_init_obs_fn: Optional[cmrl.types.ResetFnType] = None,
                 generator: Optional[np.random.Generator] = None,
                 penalty_coeff: float = .0

                 ):
        self.dynamics = dynamics
        if generator:
            self.generator = generator
        else:
            self.generator = self._np_random

        self.penalty_coeff = penalty_coeff

        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        assert self.dynamics.learned_reward or reward_fn
        assert self.dynamics.learned_termination or termination_fn
        self.learned_reward = self.dynamics.learned_reward
        self.learned_termination = self.dynamics.learned_termination
        self.get_batch_init_obs_fn = get_init_obs_fn

        self.device = dynamics.device

        self._current_batch_obs: Optional[torch.Tensor] = None

        self.completed = True

    def step(self, action):
        assert self.completed, "fake-env not completed"
        batch_next_obs, batch_reward, batch_terminal = self.batch_step(action.reshape(1, action.shape[0]))
        return batch_next_obs[0], batch_reward[0], batch_terminal[0], False, {}

    @staticmethod
    def get_penalty(ensemble_batch_next_obs):
        avg = np.mean(ensemble_batch_next_obs, axis=0)  # average predictions over models
        diffs = ensemble_batch_next_obs - avg
        dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
        penalty = np.max(dists, axis=0)  # max distances over models

        return penalty

    def batch_step(
            self,
            batch_action: cmrl.types.TensorType,
            deterministic: bool = False):
        assert self.completed, "fake-env not completed"
        assert len(batch_action.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(batch_action, np.ndarray):
                batch_action = torch.from_numpy(batch_action).to(self.device)
            dynamics_pred = self.dynamics.query(self._current_batch_obs, batch_action, return_as_np=True)

            # transition
            batch_next_obs = self.get_dynamics_predict(dynamics_pred, "transition", deterministic=deterministic)
            if self.learned_reward:
                batch_reward = self.get_dynamics_predict(dynamics_pred, "reward_mech", deterministic=deterministic)
            else:
                batch_reward = self.reward_fn(batch_next_obs, self._current_batch_obs, batch_action)
            if self.learned_termination:
                batch_terminal = self.get_dynamics_predict(dynamics_pred, "termination_mech",
                                                           deterministic=deterministic)
            else:
                batch_terminal = self.termination_fn(batch_next_obs, self._current_batch_obs, batch_action)

            if self.penalty_coeff != 0:
                batch_reward -= self.get_penalty(dynamics_pred['batch_next_obs']["mean"]).reshape(
                    batch_reward.shape) * self.penalty_coeff

            self._current_batch_obs = batch_next_obs
            return batch_next_obs, batch_reward, batch_terminal

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        if not self.completed:
            pass
        else:
            init_obs = self.get_batch_init_obs_fn(1).astype(np.float32)
            self.batch_reset(init_obs)
            return init_obs[0]

    def batch_reset(self, init_batch_obs):
        assert self.completed, "fake-env not completed"
        assert len(init_batch_obs.shape) == 2  # batch, obs_dim

        self._current_batch_obs = init_batch_obs

    def render(self, mode="human"):
        raise NotImplementedError

    def get_dynamics_predict(self,
                             origin_predict: Dict,
                             mech: str,
                             deterministic: bool = False, ):
        variable = self.dynamics.get_variable_by_mech(mech)
        ensemble_mean, ensemble_logvar = origin_predict[variable]["mean"], origin_predict[variable]["logvar"]
        batch_size = ensemble_mean.shape[1]
        random_index = getattr(self.dynamics, mech).get_random_index(batch_size, self.generator)
        if deterministic:
            pred = ensemble_mean[random_index, np.arange(batch_size)]
        else:
            ensemble_std = np.sqrt(np.exp(ensemble_logvar))
            pred = ensemble_mean[random_index, np.arange(batch_size)] + \
                   ensemble_std[random_index, np.arange(batch_size)] * \
                   self.generator.normal(size=ensemble_mean.shape[1:]).astype(np.float32)
        return pred


class FakeEnv:
    def __init__(
            self,
            env: gym.Env,
            dynamics: BaseDynamics,
            reward_fn: Optional[cmrl.types.RewardFnType] = None,
            termination_fn: Optional[cmrl.types.TermFnType] = None,
            generator: Optional[np.random.Generator] = None,
            penalty_coeff=0.,
            penalty_learned_var=False,
    ):
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        assert self.dynamics.learned_reward or reward_fn
        assert self.dynamics.learned_termination or termination_fn
        self.learned_reward = self.dynamics.learned_reward
        self.learned_termination = self.dynamics.learned_termination

        self.penalty_coeff = penalty_coeff
        self.penalty_learned_var = penalty_learned_var

        self.device = dynamics.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_batch_obs: Optional[torch.Tensor] = None
        if generator:
            self.generator = generator
        else:
            self.generator = np.random.default_rng()
        self._return_as_np = True

    def reset(
            self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ):
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        self._current_batch_obs = initial_obs_batch
        self._return_as_np = return_as_np

    def get_dynamics_predict(self,
                             origin_predict: Dict,
                             mech: str,
                             deterministic: bool = False, ):
        variable = self.dynamics.get_variable_by_mech(mech)
        ensemble_mean, ensemble_logvar = origin_predict[variable]["mean"], origin_predict[variable]["logvar"]
        batch_size = ensemble_mean.shape[1]
        random_index = getattr(self.dynamics, mech).get_random_index(batch_size, self.generator)
        if deterministic:
            pred = ensemble_mean[random_index, np.arange(batch_size)]
        else:
            ensemble_std = np.sqrt(np.exp(ensemble_logvar))
            pred = ensemble_mean[random_index, np.arange(batch_size)] + \
                   ensemble_std[random_index, np.arange(batch_size)] * \
                   self.generator.normal(size=ensemble_mean.shape[1:]).astype(np.float32)

        avg_ensemble_mean = np.mean(ensemble_mean, axis=0)  # average predictions over models
        diffs = ensemble_mean - avg_ensemble_mean
        dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
        penalty = np.max(dists, axis=0)  # max distances over models
        return pred, penalty

    def step(
            self,
            batch_action: cmrl.types.TensorType,
            deterministic: bool = False):
        assert len(batch_action.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(batch_action, np.ndarray):
                batch_action = torch.from_numpy(batch_action).to(self.device)
            dynamics_pred = self.dynamics.query(self._current_batch_obs, batch_action, return_as_np=True)

            # transition
            batch_next_obs, penalty = self.get_dynamics_predict(dynamics_pred, "transition",
                                                                deterministic=deterministic)

            if self.learned_reward:
                batch_reward, _ = self.get_dynamics_predict(dynamics_pred, "reward_mech", deterministic=deterministic)
            else:
                batch_reward = self.reward_fn(batch_next_obs, self._current_batch_obs, batch_action)
            if self.learned_termination:
                batch_terminal, _ = self.get_dynamics_predict(dynamics_pred, "termination_mech",
                                                              deterministic=deterministic)
            else:
                batch_terminal = self.termination_fn(batch_next_obs, self._current_batch_obs, batch_action)

            if self.penalty_coeff != 0:
                batch_reward -= penalty.reshape(batch_reward.shape) * self.penalty_coeff

            self._current_batch_obs = batch_next_obs
            return batch_next_obs, batch_reward, batch_terminal


class GymBehaviouralFakeEnv(gym.Env):
    def __init__(self,
                 fake_env,
                 real_env):
        self.fake_env = fake_env
        self.real_env = real_env

        self.current_obs = None

        self.action_space = self.real_env.action_space
        self.observation_space = self.real_env.observation_space

    def step(self, action):
        batch_next_obs, batch_reward, batch_terminal = self.fake_env.step(np.array([action], dtype=np.float32),
                                                                          deterministic=True)
        self.current_obs = batch_next_obs[0]
        return batch_next_obs[0], batch_reward[0][0], batch_terminal[0][0], False, {}

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
        obs = self.real_env.reset()
        self.current_obs = obs
        self.fake_env.reset(np.array([obs], dtype=np.float32))
        return obs
