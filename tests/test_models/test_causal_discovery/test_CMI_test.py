import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from cmrl.models.causal_discovery.CMI_test import TransitionConditionalMutualInformationTest


class TestTransitionConditionalMutualInformationTest(TestCase):
    def setUp(self) -> None:
        self.obs_size = 11
        self.action_size = 3
        self.num_layers = 4
        self.ensemble_num = 7
        self.hid_size = 200
        self.batch_size = 128
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cmi_test = TransitionConditionalMutualInformationTest(
            obs_size=self.obs_size,
            action_size=self.action_size,
            device=self.device,
            num_layers=self.num_layers,
            ensemble_num=self.ensemble_num,
            hid_size=self.hid_size,
        )
        self.batch_obs = torch.rand([self.ensemble_num, self.batch_size, self.obs_size]).to(self.device)
        self.batch_action = torch.rand([self.ensemble_num, self.batch_size, self.action_size]).to(self.device)

    def test_parallel_deterministic_forward(self):
        parallel_batch_obs = torch.rand(
            [self.obs_size + self.action_size + 1, self.ensemble_num, self.batch_size, self.obs_size]
        ).to(self.device)
        mean, logvar = self.cmi_test.forward(parallel_batch_obs, self.batch_action)
        assert (
            mean.shape
            == logvar.shape
            == (self.obs_size + self.action_size + 1, self.ensemble_num, self.batch_size, self.obs_size)
        )

    def test_gaussian_forward(self):
        mean, logvar = self.cmi_test.forward(self.batch_obs, self.batch_action)
        assert (
            mean.shape
            == logvar.shape
            == (self.obs_size + self.action_size + 1, self.ensemble_num, self.batch_size, self.obs_size)
        )

    def test_load(self):
        tempdir = Path(tempfile.gettempdir())
        model_dir = tempdir / "temp_model"
        if not model_dir.exists():
            model_dir.mkdir()

        mean, logvar = self.cmi_test.forward(self.batch_obs, self.batch_action)
        self.cmi_test.save(model_dir)

        new_cmi_test = TransitionConditionalMutualInformationTest(
            obs_size=self.obs_size,
            action_size=self.action_size,
            device=self.device,
            num_layers=self.num_layers,
            ensemble_num=self.ensemble_num,
            hid_size=self.hid_size,
        )

        new_mean, new_logvar = new_cmi_test.forward(self.batch_obs, self.batch_action)
        assert not (mean == new_mean).all()

        new_cmi_test.load(model_dir)
        new_mean, new_logvar = new_cmi_test.forward(self.batch_obs, self.batch_action)
        assert (mean == new_mean).all()
