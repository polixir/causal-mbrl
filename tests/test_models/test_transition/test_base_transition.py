import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from cmrl.models.transition.base_transition import BaseEnsembleTransition


class TestBasicEnsembleGaussianMLP(TestCase):
    def setUp(self) -> None:
        self.obs_size = 11
        self.action_size = 3
        self.ensemble_num = 7
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deterministic_transition = BaseEnsembleTransition(
            obs_size=self.obs_size,
            action_size=self.action_size,
            device=self.device,
            ensemble_num=self.ensemble_num,
            deterministic=True,
        )

    def test_build(self):
        assert True
