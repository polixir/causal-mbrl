from typing import Optional, List, Dict, Union, MutableMapping
from abc import abstractmethod

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from stable_baselines3.common.logger import Logger

from cmrl.utils.types import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable
from cmrl.models.networks.base_network import BaseNetwork
from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder
from cmrl.models.util import create_decoders, create_encoders


class BaseCausalMech:
    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        node_dim: int,
        variable_encoders: Optional[Dict[str, VariableEncoder]],
        variable_decoders: Optional[Dict[str, VariableDecoder]],
        ensemble_num: int = 7,
        elite_num: int = 5,
        # forward method
        residual: bool = True,
        multi_step: str = "none",
        # trainer
        optim_lr: float = 1e-4,
        optim_weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        optim_encoder: bool = True,
        # logger
        logger: Optional[Logger] = None,
        # others
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        self.name = name
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.node_dim = node_dim
        self.variable_encoders = variable_encoders
        self.variable_decoders = variable_decoders
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        # forward method
        self.residual = residual
        self.multi_step = multi_step
        # trainer
        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.optim_eps = optim_eps
        self.optim_encoder = optim_encoder
        # logger
        self.logger = logger
        # others
        self.device = device

        self.input_var_num = len(self.input_variables)
        self.output_var_num = len(self.output_variables)

        if self.variable_encoders is None:
            assert self.optim_encoder
            self.variable_encoders = create_encoders(input_variables, node_dim=self.node_dim, device=self.device)
        if self.variable_decoders is None:
            self.variable_decoders = create_decoders(output_variables, node_dim=self.node_dim, device=self.device)
        self.check_coder()

        self.network: Optional[BaseNetwork] = None
        self.graph: Optional[BaseGraph] = None

        self.build_network()
        self.build_graph()

        self.total_epoch = 0
        self.elite_indices: List[int] = []

    def check_coder(self):
        assert len(self.input_variables) == len(self.variable_encoders)
        assert len(self.output_variables) == len(self.variable_decoders)

        for var in self.input_variables:
            assert var.name in self.variable_encoders
            encoder = self.variable_encoders[var.name]
            assert encoder.node_dim == self.node_dim

        for var in self.output_variables:
            assert var.name in self.variable_decoders
            decoder = self.variable_decoders[var.name]
            assert decoder.node_dim == self.node_dim

    @abstractmethod
    def forward(self, inputs: MutableMapping[str, Union[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

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

    @abstractmethod
    def build_graph(self):
        raise NotImplementedError

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


# Causal = TypeVar("Causal", bound=BaseCausalMech)
#
#
# class BaseMultiStepCausalMech(BaseCausalMech):
#     def __init__(
#         self,
#         single_step_mech_class: Type[Causal],
#         input_variables: List[Variable],
#         output_variables: List[Variable],
#         node_dim: int,
#         variable_encoders: Dict[str, VariableEncoder],
#         variable_decoders: Dict[str, VariableDecoder],
#         **kwargs
#     ):
#         super(BaseMultiStepCausalMech, self).__init__(
#             input_variables=input_variables,
#             output_variables=output_variables,
#             node_dim=node_dim,
#             variable_encoders=variable_encoders,
#             variable_decoders=variable_decoders,
#         )
#
#         self.single_step_mech = single_step_mech_class(**kwargs)
#         pass
#
#     @abstractmethod
#     def build_network(self):
#         raise NotImplementedError
#
#     @abstractmethod
#     def build_graph(self):
#         raise NotImplementedError
