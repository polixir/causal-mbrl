from typing import Optional, List, Dict, Union, MutableMapping
from abc import abstractmethod, ABC
import pathlib

import torch
from torch.utils.data import DataLoader

from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.graphs.binary_graph import BinaryGraph
from cmrl.utils.variables import Variable


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
        self.graph: Optional[BaseGraph] = None
        self.discovery: bool = True

    @abstractmethod
    def learn(self, train_loader: DataLoader, valid_loader: DataLoader, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @property
    def causal_graph(self) -> torch.Tensor:
        """property causal graph"""
        if self.graph is None:
            return torch.ones(len(self.input_variables), len(self.output_variables), dtype=torch.int, device=self.device)
        else:
            return self.graph.get_binary_adj_matrix()

    @property
    def forward_mask(self) -> torch.Tensor:
        """property input masks"""
        return self.causal_graph.T

    def set_oracle_graph(self, graph_data):
        self.discovery = False
        self.graph = BinaryGraph(self.input_var_num, self.output_var_num, device=self.device)
        self.graph.set_data(graph_data=graph_data)

    def save(self):
        pass

    def load(self, load_dir: Union[str, pathlib.Path]):
        pass
