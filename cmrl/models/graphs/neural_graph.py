import pathlib
from typing import Optional, Union, Tuple

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from cmrl.models.graphs.base_graph import BaseGraph

default_network_cfg = DictConfig(
    dict(
        _target_="cmrl.models.networks.ParallelMLP",
        _partial_=True,
        _recursive_=False,
        hidden_dims=[200, 200],
        bias=True,
        activation_fn_cfg=dict(_target_="torch.nn.ReLU"),
    )
)


class NeuralGraph(BaseGraph):

    _MASK_VALUE = 0

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        extra_dim: Optional[Union[int, Tuple[int]]] = None,
        include_input: bool = False,
        network_cfg: Optional[DictConfig] = default_network_cfg,
        device: Union[str, torch.device] = "cpu",
        *args,
        **kwargs
    ) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim, extra_dim=extra_dim, include_input=include_input, *args, **kwargs)

        self._network_cfg = network_cfg
        self.device = device

        self._build_graph_network()

    def _build_graph_network(self):
        """called at the last of ``NeuralGraph.__init__``"""
        network_extra_dims = self._extra_dim
        if isinstance(network_extra_dims, int):
            network_extra_dims = [network_extra_dims]

        self.graph = instantiate(self._network_cfg)(
            input_dim=self._in_dim,
            output_dim=self._in_dim * self._out_dim,
            extra_dims=network_extra_dims,
        ).to(self.device)

    @property
    def parameters(self) -> Tuple[torch.Tensor]:
        return tuple(self.graph.parameters())

    def get_adj_matrix(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        adj_mat = self.graph(inputs)
        adj_mat = adj_mat.reshape(*adj_mat.shape[:-1], self._in_dim, self._out_dim)

        if self._include_input:
            adj_mat[..., torch.arange(self._in_dim), torch.arange(self._in_dim)] = self._MASK_VALUE

        return adj_mat

    def get_binary_adj_matrix(self, inputs: torch.Tensor, threshold: float, *args, **kwargs) -> torch.Tensor:
        return (self.get_adj_matrix(inputs) > threshold).int()

    def save(self, save_dir: Union[str, pathlib.Path]):
        torch.save({"graph_network": self.graph.state_dict()}, pathlib.Path(save_dir) / "graph.pth")

    def load(self, load_dir: Union[str, pathlib.Path]):
        data_dict = torch.load(pathlib.Path(load_dir) / "graph.pth", map_location=self.device)
        self.graph.load_state_dict(data_dict["graph_network"])
