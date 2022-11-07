from typing import Optional, List, Dict, Union, MutableMapping
from abc import abstractmethod, ABC
from itertools import chain
import pathlib
import itertools
import copy

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer
from stable_baselines3.common.logger import Logger
from omegaconf import DictConfig
from hydra.utils import instantiate

from cmrl.models.networks.base_network import BaseNetwork
from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.networks.coder import VariableEncoder, VariableDecoder
from cmrl.utils.variables import Variable, ContinuousVariable, DiscreteVariable, BinaryVariable


class BaseCausalMech(ABC):
    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        device: Union[str, torch.device] = "cpu",
    ):
        self.name = name
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.device = device

        self.input_var_num = len(self.input_variables)
        self.output_var_num = len(self.output_variables)

    @abstractmethod
    def learn(self, train_loader: DataLoader, valid_loader: DataLoader, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
