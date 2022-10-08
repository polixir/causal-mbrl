import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from cmrl.models.transition.multi_step.forward_euler import ForwardEulerTransition
from cmrl.models.transition.one_step.plain_ensemble import (
    PlainEnsembleGaussianTransition,
)


class TestForwardEulerTransition(TestCase):
    def setUp(self) -> None:
        self.obs_size = 11
        self.action_size = 3
        self.num_layers = 4
        self.ensemble_num = 7
        self.hid_size = 200
        self.batch_size = 128
        self.repeat_times = 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deterministic_one_step = PlainEnsembleGaussianTransition(
            obs_size=self.obs_size,
            action_size=self.action_size,
            device=self.device,
            num_layers=self.num_layers,
            ensemble_num=self.ensemble_num,
            hid_size=self.hid_size,
            deterministic=True,
        )
        self.deterministic_transition = ForwardEulerTransition(
            one_step_transition=self.deterministic_one_step,
            repeat_times=self.repeat_times,
        )
        self.gaussian_one_step = PlainEnsembleGaussianTransition(
            obs_size=self.obs_size,
            action_size=self.action_size,
            device=self.device,
            num_layers=self.num_layers,
            ensemble_num=self.ensemble_num,
            hid_size=self.hid_size,
            deterministic=False,
        )
        self.gaussian_transition = ForwardEulerTransition(
            one_step_transition=self.gaussian_one_step, repeat_times=self.repeat_times
        )
        self.batch_obs = torch.rand(
            [self.ensemble_num, self.batch_size, self.obs_size]
        ).to(self.device)
        self.batch_action = torch.rand(
            [self.ensemble_num, self.batch_size, self.action_size]
        ).to(self.device)

    def test_deterministic_forward(self):
        mean, logvar = self.deterministic_transition.forward(
            self.batch_obs, self.batch_action
        )
        assert mean.shape == (self.ensemble_num, self.batch_size, self.obs_size)
