# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch

import cmrl.types

from cmrl.models.dynamics import BaseDynamics


class FakeEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        dynamics (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

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
        self._model_indices = None
        if generator:
            self.generator = generator
        else:
            self.generator = np.random.default_rng()
        self._return_as_np = True

    def reset(
            self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ):
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        self._current_batch_obs = initial_obs_batch
        self._return_as_np = return_as_np

    def get(self,
            dynamics_pred: Dict,
            mech: str,
            deterministic: bool = False, ):
        variable = self.dynamics.get_variable_by_mech(mech)
        ensemble_mean, ensemble_logvar = dynamics_pred[variable]["mean"], dynamics_pred[variable]["logvar"]
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
            batch_next_obs, penalty = self.get(dynamics_pred, "transition", deterministic=deterministic)

            if self.learned_reward:
                batch_reward, _ = self.get(dynamics_pred, "reward_mech", deterministic=deterministic)
            else:
                batch_reward = self.reward_fn(batch_next_obs, self._current_batch_obs, batch_action)
            if self.learned_termination:
                batch_terminal, _ = self.get(dynamics_pred, "termination_mech", deterministic=deterministic)
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
