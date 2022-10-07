from unittest import TestCase

import torch

from cmrl.models.layers import EnsembleLinearLayer, ParallelEnsembleLinearLayer


class TestParallelEnsembleLinearLayer(TestCase):
    def setUp(self) -> None:
        self.in_size = 5
        self.out_size = 6
        self.use_bias = True
        self.parallel_num = 3
        self.ensemble_num = 4
        self.batch_size = 128
        self.layer = ParallelEnsembleLinearLayer(
            in_size=self.in_size,
            out_size=self.out_size,
            use_bias=self.use_bias,
            parallel_num=self.parallel_num,
            ensemble_num=self.ensemble_num,
        )

    def test_forward(self):
        model_in = torch.rand(
            (self.parallel_num, self.ensemble_num, self.batch_size, self.in_size)
        )
        model_out = self.layer(model_in)
        assert model_out.shape == (
            self.parallel_num,
            self.ensemble_num,
            self.batch_size,
            self.out_size,
        )


class TestEnsembleLinearLayer(TestCase):
    def setUp(self) -> None:
        self.in_size = 5
        self.out_size = 6
        self.use_bias = True
        self.ensemble_num = 4
        self.batch_size = 128
        self.layer = EnsembleLinearLayer(
            in_size=self.in_size,
            out_size=self.out_size,
            use_bias=self.use_bias,
            ensemble_num=self.ensemble_num,
        )

    def test_forward(self):
        model_in = torch.rand((self.ensemble_num, self.batch_size, self.in_size))
        model_out = self.layer(model_in)
        assert model_out.shape == (self.ensemble_num, self.batch_size, self.out_size)
