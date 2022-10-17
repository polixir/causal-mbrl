# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import warnings
from typing import Any, List, Optional, Sequence, Sized, Tuple, Type, Union

import numpy as np

from cmrl.types import InteractionBatch


def _consolidate_batches(batches: Sequence[InteractionBatch]) -> InteractionBatch:
    len_batches = len(batches)
    b0 = batches[0]
    obs = np.empty((len_batches,) + b0.batch_obs.shape, dtype=b0.batch_obs.dtype)
    act = np.empty((len_batches,) + b0.batch_action.shape, dtype=b0.batch_action.dtype)
    next_obs = np.empty((len_batches,) + b0.batch_obs.shape, dtype=b0.batch_obs.dtype)
    rewards = np.empty((len_batches,) + b0.batch_reward.shape, dtype=np.float32)
    dones = np.empty((len_batches,) + b0.batch_done.shape, dtype=bool)
    for i, b in enumerate(batches):
        obs[i] = b.batch_obs
        act[i] = b.batch_action
        next_obs[i] = b.batch_next_obs
        rewards[i] = b.batch_reward
        dones[i] = b.batch_done
    return InteractionBatch(obs, act, next_obs, rewards, dones)


class TransitionIterator:
    """An iterator for batches of transitions.

    The iterator can be used doing:

    .. code-block:: python

       for batch in batch_iterator:
           do_something_with_batch()

    Rather than be constructed directly, the preferred way to use objects of this class
    is for the user to obtain them from :class:`ReplayBuffer`.

    Args:
        transitions (:class:`InteractionBatch`): the transition data used to built
            the iterator.
        batch_size (int): the batch size to use when iterating over the stored data.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
    """

    def __init__(
        self,
        transitions: InteractionBatch,
        batch_size: int,
        shuffle_each_epoch: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.transitions = transitions
        self.num_stored = len(transitions)
        self._order: np.ndarray = np.arange(self.num_stored)
        self.batch_size = batch_size
        self._current_batch = 0
        self._shuffle_each_epoch = shuffle_each_epoch
        self._rng = rng if rng is not None else np.random.default_rng()

    def _get_indices_next_batch(self) -> Sized:
        start_idx = self._current_batch * self.batch_size
        if start_idx >= self.num_stored:
            raise StopIteration
        end_idx = min((self._current_batch + 1) * self.batch_size, self.num_stored)
        order_indices = range(start_idx, end_idx)
        indices = self._order[order_indices]
        self._current_batch += 1
        return indices

    def __iter__(self):
        self._current_batch = 0
        if self._shuffle_each_epoch:
            self._order = self._rng.permutation(self.num_stored)
        return self

    def __next__(self):
        return self[self._get_indices_next_batch()]

    def ensemble_size(self):
        return 0

    def __len__(self):
        return (self.num_stored - 1) // self.batch_size + 1

    def __getitem__(self, item):
        return self.transitions[item]


class BootstrapIterator(TransitionIterator):
    def __init__(
        self,
        transitions: InteractionBatch,
        batch_size: int,
        ensemble_size: int,
        shuffle_each_epoch: bool = False,
        permute_indices: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(transitions, batch_size, shuffle_each_epoch=shuffle_each_epoch, rng=rng)
        self._ensemble_size = ensemble_size
        self._permute_indices = permute_indices
        self._bootstrap_iter = ensemble_size > 1
        self.member_indices = self._sample_member_indices()

    def _sample_member_indices(self) -> np.ndarray:
        member_indices = np.empty((self.ensemble_size, self.num_stored), dtype=int)
        if self._permute_indices:
            for i in range(self.ensemble_size):
                member_indices[i] = self._rng.permutation(self.num_stored)
        else:
            member_indices = self._rng.choice(
                self.num_stored,
                size=(self.ensemble_size, self.num_stored),
                replace=True,
            )
        return member_indices

    def __iter__(self):
        super().__iter__()
        return self

    def __next__(self):
        if not self._bootstrap_iter:
            return super().__next__()
        indices = self._get_indices_next_batch()
        batches = []
        for member_idx in self.member_indices:
            content_indices = member_idx[indices]
            batches.append(self[content_indices])
        return _consolidate_batches(batches)

    def toggle_bootstrap(self):
        """Toggles whether the iterator returns a batch per model or a single batch."""
        if self.ensemble_size > 1:
            self._bootstrap_iter = not self._bootstrap_iter

    @property
    def ensemble_size(self):
        return self._ensemble_size


def _sequence_getitem_impl(
    transitions: InteractionBatch,
    batch_size: int,
    sequence_length: int,
    valid_starts: np.ndarray,
    item: Any,
):
    start_indices = valid_starts[item].repeat(sequence_length)
    increment_array = np.tile(np.arange(sequence_length), len(item))
    full_trajectory_indices = start_indices + increment_array
    return transitions[full_trajectory_indices].add_new_batch_dim(min(batch_size, len(item)))
