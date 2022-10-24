from abc import abstractmethod
import math
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn

from cmrl.models.graphs.base_graph import BaseGraph


class BaseProbGraph(BaseGraph):
    """Base class for probability modeled graphs.

    All classes derived from `BaseProbGraph` must implement the following additional methods:

        - ``sample``: sample graphs from given (or current) graph probability.
    """

    @abstractmethod
    def sample(self, graph: Optional[torch.Tensor], sample_size: Union[Tuple[int], int], *args, **kwargs) -> torch.Tensor:
        """sample from given or current graph probability.

        Args:
            graph (tensor), graph probability, use current graph parameter when given `None`.
            sample_size (tuple(int) or int), extra size of sampled graphs.

        Return:
            (tensor): [*sample_size, in_dim, out_dim] shaped multiple graphs.
        """
        pass


class BernoulliGraph(BaseProbGraph):
    """Probability (Bernoulli dist.) modeled graphs, store the graph with the
        probability parameter of the existence/orientation of edges.

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        init_param (float or torch.Tensor): initial parameter of the graph
            (sigmoid(init_param) representing the initial edge probabilities).
        device (str or torch.device): device to use for the structural parameters.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_param: Union[float, torch.Tensor] = 1e-6,
        device: Union[str, torch.device] = "cpu",
        *args,
        **kwargs
    ):
        super().__init__(in_dim, out_dim, device, *args, **kwargs)

        if isinstance(init_param, float):
            init_param = torch.ones(in_dim, out_dim) * init_param
        self.graph = nn.Parameter(init_param, requires_grad=True)

        self.to(device)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Computes the graph parameters.

        Returns:
            (tuple of tensors): all tensors representing the output
                graph (e.g. existence and orientation)
        """
        return torch.sigmoid(self.graph)

    def get_binary_graph(self, thresh: float = 0.5) -> torch.Tensor:
        """Gets the binary graph.

        Returns:
            (tensor): the binary graph tensor, shape [in_dim, out_dim];
            graph[i, j] == 1 represents i causes j
        """
        assert 0 <= thresh <= 1

        prob_graph = self()
        return prob_graph > thresh

    def sample(self, graph: Optional[torch.Tensor], sample_size: Union[Tuple[int], int], *args, **kwargs):
        """sample from given or current graph probability (Bernoulli distribution).

        Args:
            graph (tensor), graph probability, use current graph parameter when given `None`.
            sample_size (tuple(int) or int), extra size of sampled graphs.

        Return:
            (tensor): [*sample_size, in_dim, out_dim] shaped multiple graphs.
        """
        if graph is None:
            graph = self()

        if isinstance(sample_size, int):
            sample_size = (sample_size,)

        sample_prob = graph[None].expand(*sample_size, -1, -1)

        return torch.bernoulli(sample_prob)
