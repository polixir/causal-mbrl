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
            generator: Optional[torch.Generator] = None,
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
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
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
        random_index = getattr(self.dynamics, mech).get_random_index(batch_size)
        if deterministic:
            pred = ensemble_mean[random_index, np.arange(batch_size)]
        else:
            ensemble_std = np.sqrt(np.exp(ensemble_logvar))
            pred = ensemble_mean[random_index, np.arange(batch_size)] + \
                   ensemble_std[random_index, np.arange(batch_size)] * \
                   np.random.normal(size=ensemble_mean.shape[1:]).astype(np.float32)

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

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
            self,
            action_sequences: torch.Tensor,
            initial_state: np.ndarray,
            num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            assert initial_state.ndim in (1, 3)
            tiling_shape = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )
            initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            total_rewards = torch.zeros(batch_size, 1).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            for time_step in range(horizon):
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                _, rewards, dones, model_state = self.step(
                    action_batch, model_state, sample=True
                )
                rewards[terminated] = 0
                terminated |= dones
                total_rewards += rewards

            total_rewards = total_rewards.reshape(-1, num_particles)
            return total_rewards.mean(dim=1)
