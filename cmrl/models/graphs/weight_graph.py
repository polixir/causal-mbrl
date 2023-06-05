import pathlib
from typing import Optional, Union, Tuple

import torch
import numpy as np

from cmrl.models.graphs.base_graph import BaseGraph


class WeightGraph(BaseGraph):
    """Weight graph models (real graph data)

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        extra_dim (int | tuple(int) | None): extra dimensions (multi-graph).
        include_input (bool): whether inlcude input variables in the output variables.
        init_param (int | Tensor | ndarray): initial parameter of the weight graph。
        requires_grad (bool): whether the graph parameters require gradient computation.
        device (str or torch.device): device to use for the graph parameters.
    """

    _MASK_VALUE = 0

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        extra_dim: Optional[Union[int, Tuple[int]]] = None,
        include_input: bool = False,
        init_param: Union[float, torch.Tensor, np.ndarray] = 1.0,
        requires_grad: bool = False,
        device: Union[str, torch.device] = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(in_dim, out_dim, extra_dim, include_input, *args, **kwargs)
        self._requires_grad = requires_grad

        graph_size = (in_dim, out_dim)
        if extra_dim is not None:
            if isinstance(extra_dim, int):
                extra_dim = (extra_dim,)
            graph_size = extra_dim + graph_size

        if isinstance(init_param, float):
            self.graph = torch.ones(graph_size, dtype=torch.float32, device=device) * init_param
        else:
            assert (
                init_param.shape == graph_size
            ), f"initial parameters shape mismatch (given {init_param.shape}, while {graph_size} required)"
            self.graph = torch.as_tensor(init_param, dtype=torch.float32, device=device)

        if requires_grad:
            self.graph.requires_grad_()

        # remove self loop
        if self._include_input:
            with torch.no_grad():
                self.graph[..., torch.arange(self._in_dim), torch.arange(self._in_dim)] = self._MASK_VALUE

        self.device = device

    @property
    def parameters(self) -> Tuple[torch.Tensor]:
        return (self.graph,)

    @property
    def requries_grad(self) -> bool:
        return self._requires_grad

    def get_adj_matrix(self, *args, **kwargs) -> torch.Tensor:
        return self.graph

    def get_binary_adj_matrix(self, threshold: float, *args, **kwargs) -> torch.Tensor:
        return (self.get_adj_matrix() > threshold).int()

    @torch.no_grad()
    def set_data(self, graph_data: Union[torch.Tensor, np.ndarray]):
        assert (
            self.graph.shape == graph_data.shape
        ), f"graph data shape mismatch (given {graph_data.shape}, while {self.graph.shape} required)"
        self.graph.data = torch.as_tensor(graph_data, dtype=torch.float32, device=self.device)

        # remove self loop
        if self._include_input:
            self.graph[..., torch.arange(self._in_dim), torch.arange(self._in_dim)] = self._MASK_VALUE

    def save(self, save_dir: Union[str, pathlib.Path]):
        torch.save({"graph_data": self.graph}, pathlib.Path(save_dir) / "graph.pth")

    def load(self, load_dir: Union[str, pathlib.Path]):
        data_dict = torch.load(pathlib.Path(load_dir) / "graph.pth", map_location=self.device)
        self.graph = data_dict["graph_data"]
