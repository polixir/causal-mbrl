import pathlib
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate

from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.graphs.prob_graph import BaseProbGraph

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


class NeuralBernoulliGraph(NeuralGraph, BaseProbGraph):

    _MASK_VALUE = -9e15

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
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            extra_dim=extra_dim,
            include_input=include_input,
            network_cfg=network_cfg,
            device=device,
            *args,
            **kwargs
        )

    def _build_graph_network(self):
        super()._build_graph_network()

        def init_weights_zero(layer):
            for pname, params in layer.named_parameters():
                if "weight" in pname:
                    nn.init.zeros_(params)

        self.graph.apply(init_weights_zero)

    def get_adj_matrix(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.sigmoid(super().get_adj_matrix(inputs, *args, **kwargs))

    def get_binary_adj_matrix(self, inputs: torch.Tensor, threshold: float = 0.5, *args, **kwargs) -> torch.Tensor:
        """return the binary adjacency matrices corresponding to the inputs (w/o grad.)"""
        return super().get_binary_adj_matrix(inputs, threshold, *args, **kwargs)

    def sample(
        self,
        prob_matrix: Optional[torch.Tensor],
        sample_size: Union[Tuple[int], int],
        reparameterization: Optional[str] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """sample from given or current graph probability (Bernoulli distribution).

        Args:
            prob_matrix (tensor), graph probability, can not be empty here.
            sample_size (tuple(int) or int), extra size of sampled graphs.

        Return:
            (tensor): [*sample_size, *extra_dim, in_dim, out_dim] shaped multiple graphs.
        """
        if prob_matrix is None:
            raise ValueError("Porb. matrix can not be empty")

        if isinstance(sample_size, int):
            sample_size = (sample_size,)

        sample_prob = prob_matrix[None].expand(*sample_size, -1, -1)

        if reparameterization is None:
            return torch.bernoulli(sample_prob)
        elif reparameterization == "gumbel-softmax":
            return F.gumbel_softmax(torch.stack((sample_prob, 1 - sample_prob)), hard=True, dim=0)[0]
        else:
            raise NotImplementedError

    def sample_from_inputs(
        self,
        inputs: torch.Tensor,
        sample_size: Union[Tuple[int], int],
        reparameterization: Optional[str] = "gumbel-softmax",
        *args,
        **kwargs
    ) -> torch.Tensor:
        """sample adjacency matrix from inputs (genereated Bernoulli distribution given the inputs).

        Args:
            inputs (tensor), input samples.
            sample_size (tuple(int) or int), extra size of sampled graphs.

        Return:
            (tensor): [*sample_size, *extra_dim, in_dim, out_dim] shaped multiple graphs.
        """
        if isinstance(sample_size, int):
            sample_size = (sample_size,)

        inputs = inputs[None].expand(*sample_size, *((-1,) * len(inputs.shape)))
        sample_prob = self.get_adj_matrix(inputs)

        if reparameterization is None:
            return torch.bernoulli(sample_prob)
        elif reparameterization == "gumbel-softmax":
            return F.gumbel_softmax(torch.stack((sample_prob, 1 - sample_prob)), hard=True, dim=0)[0]
        else:
            raise NotImplementedError
