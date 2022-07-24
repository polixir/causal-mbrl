from unittest import TestCase

from cmrl.models.transition.one_step.external_mask_ensemble import ExternalMaskEnsembleGaussianTransition
import torch
import tempfile
from pathlib import Path


class TestExternalMaskEnsembleGaussianTransition(TestCase):
    def setUp(self) -> None:
        self.obs_size = 11
        self.action_size = 3
        self.num_layers = 4
        self.ensemble_num = 7
        self.hid_size = 200
        self.batch_size = 128
        self.device = "cuda"
        self.deterministic_transition = ExternalMaskEnsembleGaussianTransition(obs_size=self.obs_size,
                                                                               action_size=self.action_size,
                                                                               device=self.device,
                                                                               num_layers=self.num_layers,
                                                                               ensemble_num=self.ensemble_num,
                                                                               hid_size=self.hid_size,
                                                                               deterministic=True)
        self.gaussian_transition = ExternalMaskEnsembleGaussianTransition(obs_size=self.obs_size,
                                                                          action_size=self.action_size,
                                                                          device=self.device,
                                                                          num_layers=self.num_layers,
                                                                          ensemble_num=self.ensemble_num,
                                                                          hid_size=self.hid_size,
                                                                          deterministic=False)
        self.batch_obs = torch.rand([self.ensemble_num, self.batch_size, self.obs_size]).to(self.device)
        self.batch_action = torch.rand([self.ensemble_num, self.batch_size, self.action_size]).to(self.device)

    def test_deterministic_forward(self):
        input_mask = torch.ones(self.obs_size, self.obs_size + self.action_size)
        self.deterministic_transition.set_input_mask(input_mask)
        mean, logvar = self.deterministic_transition.forward(self.batch_obs, self.batch_action)
        assert mean.shape == (self.ensemble_num, self.batch_size, self.obs_size) and logvar is None

    def test_gaussian_forward(self):
        input_mask = torch.ones(self.obs_size, self.obs_size + self.action_size)
        self.gaussian_transition.set_input_mask(input_mask)
        mean, logvar = self.gaussian_transition.forward(self.batch_obs, self.batch_action)
        assert mean.shape == logvar.shape == (self.ensemble_num, self.batch_size, self.obs_size)

    def test_mask_input(self):
        input_mask = torch.ones(self.obs_size, self.obs_size + self.action_size)
        self.gaussian_transition.set_input_mask(input_mask)
        mean, logvar = self.gaussian_transition.forward(self.batch_obs, self.batch_action)

        new_input_mask = input_mask.clone()
        new_input_mask[0] = torch.zeros(self.obs_size + self.action_size)
        self.gaussian_transition.set_input_mask(new_input_mask)
        new_mean, new_logvar = self.gaussian_transition.forward(self.batch_obs, self.batch_action)
        assert not (mean == new_mean).all()
        assert (mean[..., 1:] == new_mean[..., 1:]).all()

    def test_load(self):
        tempdir = Path(tempfile.gettempdir())
        model_dir = tempdir / "temp_model"
        if not model_dir.exists():
            model_dir.mkdir()

        input_mask = torch.ones(self.obs_size, self.obs_size + self.action_size)
        self.gaussian_transition.set_input_mask(input_mask)
        mean, logvar = self.gaussian_transition.forward(self.batch_obs, self.batch_action)
        self.gaussian_transition.save(model_dir)

        new_gaussian_transition = ExternalMaskEnsembleGaussianTransition(obs_size=self.obs_size,
                                                                         action_size=self.action_size,
                                                                         device=self.device,
                                                                         num_layers=self.num_layers,
                                                                         ensemble_num=self.ensemble_num,
                                                                         hid_size=self.hid_size,
                                                                         deterministic=False)

        new_gaussian_transition.set_input_mask(input_mask)
        new_mean, new_logvar = new_gaussian_transition.forward(self.batch_obs, self.batch_action)
        assert not (mean == new_mean).all()

        new_gaussian_transition.load(model_dir)
        new_mean, new_logvar = new_gaussian_transition.forward(self.batch_obs, self.batch_action)
        assert (mean == new_mean).all()
