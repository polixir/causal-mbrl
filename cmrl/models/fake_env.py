# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, List, Optional, Sequence, Type, Union, Dict

import gym
from gym.core import ObsType, ActType
import numpy as np
import torch

import cmrl.types
from cmrl.models.dynamics import BaseDynamics

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


class VecFakeEnv(VecEnv):
    def __init__(self,
                 num_envs: int,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space):
        super(VecFakeEnv, self).__init__(num_envs=num_envs,
                                         observation_space=observation_space,
                                         action_space=action_space)

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

        self._envs_length = [0 for _ in range(self.num_envs)]

    def set_up(self,
               dynamics: BaseDynamics,
               reward_fn: Optional[cmrl.types.RewardFnType] = None,
               termination_fn: Optional[cmrl.types.TermFnType] = None,
               get_init_obs_fn: Optional[cmrl.types.ResetFnType] = None,
               penalty_coeff: float = .0,
               deterministic=False,
               max_episode_steps=1000
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
                batch_terminal = self.get_dynamics_predict(dynamics_pred, "termination_mech",
                                                           deterministic=self.deterministic)
            else:
                batch_terminal = self.termination_fn(batch_next_obs, self._current_batch_obs,
                                                     self._current_batch_action)

            if self.penalty_coeff != 0:
                batch_reward -= self.get_penalty(dynamics_pred['batch_next_obs']["mean"]).reshape(
                    batch_reward.shape) * self.penalty_coeff

        self._current_batch_obs = batch_next_obs.copy()
        batch_reward = batch_reward.reshape(self.num_envs)
        batch_terminal = batch_terminal.reshape(self.num_envs)
        infos = [{} for _ in range(self.num_envs)]
        for idx in range(self.num_envs):
            self._envs_length[idx] += 1
            batch_terminal[idx] = batch_terminal[idx] or self._envs_length[idx] >= self.max_episode_steps
            if batch_terminal[idx]:
                self.single_reset(idx)
                infos[idx]["terminal_observation"] = batch_next_obs[idx]

        return self._current_batch_obs.copy(), batch_reward.copy(), batch_terminal.copy(), infos

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        if self.has_set_up:
            self._current_batch_obs = self.get_batch_init_obs_fn(self.num_envs)
            self._envs_length = [0 for _ in range(self.num_envs)]
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

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass


if __name__ == '__main__':
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: gym.make("CartPole-v1") for i in range(16)])
    env.reset()
    print(env.step(np.random.randint(0, 1, 16)))

#
# class FakeEnv:
#     def __init__(
#             self,
#             env: gym.Env,
#             dynamics: BaseDynamics,
#             reward_fn: Optional[cmrl.types.RewardFnType] = None,
#             termination_fn: Optional[cmrl.types.TermFnType] = None,
#             generator: Optional[np.random.Generator] = None,
#             penalty_coeff=0.,
#             penalty_learned_var=False,
#     ):
#         self.dynamics = dynamics
#         self.reward_fn = reward_fn
#         self.termination_fn = termination_fn
#         assert self.dynamics.learned_reward or reward_fn
#         assert self.dynamics.learned_termination or termination_fn
#         self.learned_reward = self.dynamics.learned_reward
#         self.learned_termination = self.dynamics.learned_termination
#
#         self.penalty_coeff = penalty_coeff
#         self.penalty_learned_var = penalty_learned_var
#
#         self.device = dynamics.device
#
#         self.observation_space = env.observation_space
#         self.action_space = env.action_space
#
#         self._current_batch_obs: Optional[torch.Tensor] = None
#         if generator:
#             self.generator = generator
#         else:
#             self.generator = np.random.default_rng()
#         self._return_as_np = True
#
#     def reset(
#             self, initial_obs_batch: np.ndarray, return_as_np: bool = True
#     ):
#         assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
#         self._current_batch_obs = initial_obs_batch
#         self._return_as_np = return_as_np
#
#     def get_dynamics_predict(self,
#                              origin_predict: Dict,
#                              mech: str,
#                              deterministic: bool = False, ):
#         variable = self.dynamics.get_variable_by_mech(mech)
#         ensemble_mean, ensemble_logvar = origin_predict[variable]["mean"], origin_predict[variable]["logvar"]
#         batch_size = ensemble_mean.shape[1]
#         random_index = getattr(self.dynamics, mech).get_random_index(batch_size, self.generator)
#         if deterministic:
#             pred = ensemble_mean[random_index, np.arange(batch_size)]
#         else:
#             ensemble_std = np.sqrt(np.exp(ensemble_logvar))
#             pred = ensemble_mean[random_index, np.arange(batch_size)] + \
#                    ensemble_std[random_index, np.arange(batch_size)] * \
#                    self.generator.normal(size=ensemble_mean.shape[1:]).astype(np.float32)
#
#         avg_ensemble_mean = np.mean(ensemble_mean, axis=0)  # average predictions over models
#         diffs = ensemble_mean - avg_ensemble_mean
#         dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
#         penalty = np.max(dists, axis=0)  # max distances over models
#         return pred, penalty
#
#     def step(
#             self,
#             self._current_batch_action: cmrl.types.TensorType,
#             deterministic: bool = False):
#         assert len(self._current_batch_action.shape) == 2  # batch, action_dim
#         with torch.no_grad():
#             # if actions is tensor, code assumes it's already on self.device
#             if isinstance(self._current_batch_action, np.ndarray):
#                 self._current_batch_action = torch.from_numpy(self._current_batch_action).to(self.device)
#             dynamics_pred = self.dynamics.query(self._current_batch_obs, self._current_batch_action, return_as_np=True)
#
#             # transition
#             batch_next_obs, penalty = self.get_dynamics_predict(dynamics_pred, "transition",
#                                                                 deterministic=deterministic)
#
#             if self.learned_reward:
#                 batch_reward, _ = self.get_dynamics_predict(dynamics_pred, "reward_mech", deterministic=deterministic)
#             else:
#                 batch_reward = self.reward_fn(batch_next_obs, self._current_batch_obs, self._current_batch_action)
#             if self.learned_termination:
#                 batch_terminal, _ = self.get_dynamics_predict(dynamics_pred, "termination_mech",
#                                                               deterministic=deterministic)
#             else:
#                 batch_terminal = self.termination_fn(batch_next_obs, self._current_batch_obs, self._current_batch_action)
#
#             if self.penalty_coeff != 0:
#                 batch_reward -= penalty.reshape(batch_reward.shape) * self.penalty_coeff
#
#             self._current_batch_obs = batch_next_obs
#             return batch_next_obs, batch_reward, batch_terminal
#
#
# class GymBehaviouralFakeEnv(gym.Env):
#     def __init__(self,
#                  fake_env,
#                  real_env):
#         self.fake_env = fake_env
#         self.real_env = real_env
#
#         self.current_obs = None
#
#         self.action_space = self.real_env.action_space
#         self.observation_space = self.real_env.observation_space
#
#     def step(self, action):
#         batch_next_obs, batch_reward, batch_terminal = self.fake_env.step(np.array([action], dtype=np.float32),
#                                                                           deterministic=True)
#         self.current_obs = batch_next_obs[0]
#         return batch_next_obs[0], batch_reward[0][0], batch_terminal[0][0], False, {}
#
#     def render(self, mode="human"):
#         assert mode == "human"
#         self.real_env.freeze()
#         self.real_env.set_state_by_obs(self.current_obs)
#         self.real_env.render()
#         self.real_env.unfreeze()
#
#     def reset(
#             self,
#             *,
#             seed: Optional[int] = None,
#             return_info: bool = False,
#             options: Optional[dict] = None,
#     ):
#         obs = self.real_env.reset()
#         self.current_obs = obs
#         self.fake_env.reset(np.array([obs], dtype=np.float32))
#         return obs
