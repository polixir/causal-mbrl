from unittest import TestCase

import torch

from cmrl.models.reward_and_termination import BaseRewardMech


class TestBaseRewardMech(TestCase):
    def setUp(self) -> None:
        self.obs_size = 11
        self.action_size = 3
        self.num_layers = 4
        self.ensemble_num = 7
        self.hid_size = 200
        self.batch_size = 128
        self.device = "cuda"
        self.reward_mech = BaseRewardMech(
            obs_size=self.obs_size,
            action_size=self.action_size,
            device=self.device,
            num_layers=self.num_layers,
            hid_size=self.hid_size,
            deterministic=True,
        )
        self.batch_obs = torch.rand(
            [self.ensemble_num, self.batch_size, self.obs_size]
        ).to(self.device)
        self.batch_action = torch.rand(
            [self.ensemble_num, self.batch_size, self.action_size]
        ).to(self.device)

    def test_forward(self):
        mean, logvar = self.reward_mech.forward(self.batch_obs, self.batch_action)
        assert mean.shape == (self.ensemble_num, self.batch_size, 1)
