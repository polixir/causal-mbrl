from typing import Optional, List, Dict, Union, TypeVar, Type
from abc import abstractmethod

import torch

from cmrl.types import Variable, ContinuousVariable, DiscreteVariable
from cmrl.models.networks.base_network import BaseNetwork
from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder


class BaseCausalMech:
    def __init__(
        self,
        input_variables: List[Variable],
        output_variables: List[Variable],
        node_dim: int,
        variable_encoders: Dict[str, VariableEncoder],
        variable_decoders: Dict[str, VariableDecoder],
        # forward method
        residual: bool = True,
        # others
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.node_dim = node_dim
        self.variable_encoders = variable_encoders
        self.variable_decoders = variable_decoders
        self.residual = residual
        self.device = device

        self.input_var_num = len(self.input_variables)
        self.output_var_num = len(self.output_variables)

        self.check_coder()

        self.network: Optional[BaseNetwork] = None
        self.graph: Optional[BaseGraph] = None

        self.build_network()
        self.build_graph()

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
    def learn(self):
        raise NotImplementedError

    @abstractmethod
    def build_network(self):
        raise NotImplementedError

    @abstractmethod
    def build_graph(self):
        raise NotImplementedError

    def encode(self, inputs):
        pass

    def decode(self, hidden):
        pass


Causal = TypeVar("Causal", bound=BaseCausalMech)


class BaseMultiStepCausalMech(BaseCausalMech):
    def __init__(
        self,
        single_step_mech_class: Type[Causal],
        input_variables: List[Variable],
        output_variables: List[Variable],
        node_dim: int,
        variable_encoders: Dict[str, VariableEncoder],
        variable_decoders: Dict[str, VariableDecoder],
        **kwargs
    ):
        super(BaseMultiStepCausalMech, self).__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            node_dim=node_dim,
            variable_encoders=variable_encoders,
            variable_decoders=variable_decoders,
        )

        self.single_step_mech = single_step_mech_class(**kwargs)
        pass

    @abstractmethod
    def build_network(self):
        raise NotImplementedError

    @abstractmethod
    def build_graph(self):
        raise NotImplementedError
