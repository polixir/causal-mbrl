from typing import Optional, List, Dict, Union, MutableMapping
from abc import abstractmethod
from itertools import chain

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from stable_baselines3.common.logger import Logger
from omegaconf import DictConfig
from hydra.utils import instantiate

from cmrl.models.networks.base_network import BaseNetwork
from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder
from cmrl.models.util import create_decoders, create_encoders
from cmrl.utils.types import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable


class BaseCausalMech:
    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        ensemble_num: int = 7,
        elite_num: int = 5,
        # cfgs
        network_cfg: Optional[DictConfig] = None,
        encoder_cfg: Optional[DictConfig] = None,
        decoder_cfg: Optional[DictConfig] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        # forward method
        residual: bool = True,
        encoder_reduction: str = "sum",
        multi_step: str = "none",
        # logger
        logger: Optional[Logger] = None,
        # others
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        self.name = name
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        # cfgs
        self.network_cfg = network_cfg
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.optimizer_cfg = optimizer_cfg
        # forward method
        self.residual = residual
        self.encoder_reduction = encoder_reduction
        self.multi_step = multi_step
        # logger
        self.logger = logger
        # others
        self.device = device

        self.input_var_num = len(self.input_variables)
        self.output_var_num = len(self.output_variables)

        self.variable_encoders: Optional[Dict[str, VariableEncoder]] = None
        self.variable_decoders: Optional[Dict[str, VariableEncoder]] = None
        self.network: Optional[BaseNetwork] = None
        self.graph: Optional[BaseGraph] = None
        self.optimizer: Optional[Optimizer] = None

        self.build_coder()
        self.build_network()
        self.build_graph()
        self.build_optimizer()

        self.total_epoch = 0
        self.elite_indices: List[int] = []

    @abstractmethod
    def single_step_forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.multi_step.startswith("forward-euler"):
            step_num = int(self.multi_step.split()[-1])

            outputs = {}
            for step in range(step_num):
                outputs = self.single_step_forward(inputs)
                if step < step_num - 1:
                    for name in filter(lambda s: s.startswith("obs"), inputs.keys()):
                        assert inputs[name].shape[:2] == outputs["next_{}".format(name)].shape[:2]
                        assert inputs[name].shape[2] * 2 == outputs["next_{}".format(name)].shape[2]
                        inputs[name] = outputs["next_{}".format(name)][:, :, : inputs[name].shape[2]]
        else:
            raise NotImplementedError("multi-step method {} is not supported".format(self.multi_step))

        return outputs

    @abstractmethod
    def learn(
        self,
        # loader
        train_loader: DataLoader,
        valid_loader: DataLoader,
        **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def build_network(self):
        raise NotImplementedError

    def build_optimizer(self):
        assert self.network is not None, "you must build network first"
        params = (
            [self.network.parameters()]
            + [encoder.parameters() for encoder in self.variable_encoders.values()]
            + [decoder.parameters() for decoder in self.variable_decoders.values()]
        )

        self.optimizer = instantiate(self.optimizer_cfg)(params=chain(*params))

    @abstractmethod
    def build_graph(self):
        raise NotImplementedError

    def build_coder(self):
        self.variable_encoders = {}
        for var in self.input_variables:
            assert var.name not in self.variable_encoders, "duplicate name in encoders: {}".format(var.name)
            self.variable_encoders[var.name] = instantiate(self.encoder_cfg)(variable=var).to(self.device)

        assert self.decoder_input_dim

        self.variable_decoders = {}
        for var in self.output_variables:
            assert var.name not in self.variable_decoders, "duplicate name in decoders: {}".format(var.name)
            self.variable_decoders[var.name] = instantiate(self.decoder_cfg)(variable=var).to(self.device)

    def loss(self, outputs, targets):
        ensemble_num, batch_size = list(targets.values())[0].shape[:2]
        total_loss = torch.zeros(ensemble_num, batch_size, self.output_var_num)
        for i, var in enumerate(self.output_variables):
            output = outputs[var.name]
            target = targets[var.name].to(self.device)
            if isinstance(var, ContinuousVariable):
                dim = target.shape[-1]  # ensemble-num, batch-size, dim
                assert output.shape[-1] == 2 * dim
                mean, log_var = output[:, :, :dim], output[:, :, dim:]
                loss = F.gaussian_nll_loss(mean, target, log_var.exp(), reduction="none").mean(dim=-1)
                total_loss[..., i] = loss
            elif isinstance(var, DiscreteVariable):
                # TODO: onehot to int?
                raise NotImplementedError
                total_loss[..., i] = F.cross_entropy(output, target, reduction="none")
            elif isinstance(var, BinaryVariable):
                total_loss[..., i] = F.binary_cross_entropy(output, target, reduction="none")
            else:
                raise NotImplementedError
        return total_loss

    @property
    def encoder_output_dim(self):
        return self.encoder_cfg.output_dim

    @property
    def union_output_var_dim(self):
        # all output variables should be ContinuousVariable and have same variable.dim
        output_dim = []
        for var in self.output_variables:
            assert isinstance(var, ContinuousVariable), "all output variables should be ContinuousVariable"
            output_dim.append(var.dim)
        assert len(set(output_dim)) == 1, "all output variables should have same variable.dim"
        return output_dim[0]

    @property
    def decoder_input_dim(self):
        if self.decoder_cfg.identity:
            return self.union_output_var_dim * 2
        else:
            return self.decoder_cfg.input_dim

    def reduce_encoder_output(self, encoder_output: torch.Tensor) -> torch.Tensor:
        assert len(encoder_output.shape) == 4, (
            "shape of encoder_output should be (ensemble-num, batch-size, input-cvar-num, encoder-output-dim), "
            "rather than {}".format(encoder_output.shape)
        )
        if self.encoder_reduction == "sum":
            return encoder_output.sum(-2)
        elif self.encoder_reduction == "mean":
            return encoder_output.mean(-2)
        elif self.encoder_reduction == "sum":
            return encoder_output.sum(-2)
        else:
            raise NotImplementedError("not implemented encoder reduction method: {}".format(self.encoder_reduction))
