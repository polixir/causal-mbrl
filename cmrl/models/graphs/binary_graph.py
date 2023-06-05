import copy
import pathlib
from typing import Optional, Union, Tuple

import torch
import numpy as np

from cmrl.models.graphs.base_graph import BaseGraph


class BinaryGraph(BaseGraph):
    """Binary graph models (binary graph data)

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        extra_dim (int | tuple(int) | None): extra dimensions (multi-graph).
        include_input (bool): whether inlcude input variables in the output variables.
        init_param (int | Tensor | ndarray): initial parameter of the binary graph
        device (str or torch.device): device to use for the graph parameters.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        extra_dim: Optional[Union[int, Tuple[int]]] = None,
        include_input: bool = False,
        init_param: Union[int, torch.Tensor, np.ndarray] = 1,
        device: Union[str, torch.device] = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(in_dim, out_dim, extra_dim, include_input, *args, **kwargs)

        graph_size = (in_dim, out_dim)
        if extra_dim is not None:
            if isinstance(extra_dim, int):
                extra_dim = (extra_dim,)
            graph_size = extra_dim + graph_size

        if isinstance(init_param, int):
            self.graph = torch.ones(graph_size, dtype=torch.int, device=device) * int(bool(init_param))
        else:
            assert (
                init_param.shape == graph_size
            ), f"initial parameters shape mismatch (given {init_param.shape}, while {graph_size} required)"
            self.graph = torch.as_tensor(init_param, dtype=torch.bool, device=device).int()

        # remove self loop
        if self._include_input:
            self.graph[..., torch.arange(self._in_dim), torch.arange(self._in_dim)] = 0

        self.device = device

    @property
    def parameters(self) -> Tuple[torch.Tensor]:
        return (self.graph,)

    def get_adj_matrix(self, *args, **kwargs) -> torch.Tensor:
        return self.graph

    def get_binary_adj_matrix(self, *args, **kwargs) -> torch.Tensor:
        return self.get_adj_matrix()

    def set_data(self, graph_data: Union[torch.Tensor, np.ndarray]):
        assert (
            self.graph.shape == graph_data.shape
        ), f"graph data shape mismatch (given {graph_data.shape}, while {self.graph.shape} required)"
        self.graph.data = torch.as_tensor(graph_data, dtype=torch.bool, device=self.device).int()

        # remove self loop
        if self._include_input:
            self.graph[..., torch.arange(self._in_dim), torch.arange(self._in_dim)] = 0

    def save(self, save_dir: Union[str, pathlib.Path]):
        torch.save({"graph_data": self.graph}, pathlib.Path(save_dir) / "graph.pth")

    def load(self, load_dir: Union[str, pathlib.Path]):
        data_dict = torch.load(pathlib.Path(load_dir) / "graph.pth", map_location=self.device)
        self.graph = data_dict["graph_data"]
