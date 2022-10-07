import random
from unittest import TestCase

import torch

from cmrl.types import InteractionBatch


class TestTransitionBatch(TestCase):
    def setUp(self) -> None:
        self.batch_size = random.randint(0, 1000)
        self.obs_size = random.randint(0, 1000)
        self.action_size = random.randint(0, 1000)
        self.transition_batch = InteractionBatch(
            torch.rand(self.batch_size, self.obs_size),
            torch.rand(self.batch_size, self.action_size),
            torch.rand(self.batch_size, self.obs_size),
            torch.rand(self.batch_size, 1),
            torch.rand(self.batch_size, 1),
        )

    def test_as_tuple(self):
        (
            batch_obs,
            batch_action,
            batch_next_obs,
            batch_reward,
            batch_done,
        ) = self.transition_batch.as_tuple()
        assert batch_obs.shape == (self.batch_size, self.obs_size)
        assert batch_action.shape == (self.batch_size, self.action_size)
        assert batch_next_obs.shape == (self.batch_size, self.obs_size)
        assert batch_reward.shape == (self.batch_size, 1)
        assert batch_done.shape == (self.batch_size, 1)

    def test___getitem__(self):
        slice_transition = self.transition_batch[0]
        assert len(slice_transition) == self.obs_size

        new_transition = slice_transition.add_new_batch_dim(1)
        assert new_transition.batch_obs.shape == (1, self.obs_size)

    def test__get_new_shape(self):
        new_batch_size = 1
        old_shape = self.transition_batch.batch_obs.shape
        new_shape = self.transition_batch._get_new_shape(old_shape, new_batch_size)
        assert new_shape == (new_batch_size, self.batch_size, self.obs_size)
