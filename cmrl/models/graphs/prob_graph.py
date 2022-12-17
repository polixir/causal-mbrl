from abc import abstractmethod
from typing import Union, Tuple, Optional

import torch
import numpy as np
import torch.nn.functional as F

from cmrl.models.graphs.base_graph import BaseGraph
from cmrl.models.graphs.weight_graph import WeightGraph


class BaseProbGraph(BaseGraph):
    """Base class for probability modeled graphs.

    All classes derived from `BaseProbGraph` must implement the following additional methods:

        - ``sample``: sample graphs from current (or given) graph probability.
    """

    @abstractmethod
    def sample(
        self, prob_matrix: Optional[torch.Tensor], sample_size: Union[Tuple[int], int], *args, **kwargs
    ) -> torch.Tensor:
        """sample from given or current probability adjacency matrix.

        Args:
            graph (tensor), graph probability, use current graph parameter when given `None`.
            sample_size (tuple(int) or int), extra size of sampled graphs.

        Return:
            (tensor): [*sample_size, in_dim, out_dim] shaped multiple graphs.
        """
        pass


class BernoulliGraph(WeightGraph, BaseProbGraph):
    """Probability (Bernoulli dist.) graph models, store the graph with the
        probability parameter of the existence of edges.

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        extra_dim (int | tuple(int) | None): extra dimensions (multi-graph).
        include_input (bool): whether inlcude input variables in the output variables.
        init_param (int | Tensor | ndarray): initial parameter of the bernoulli graph。
        requires_grad (bool): whether the graph parameters require gradient computation.
        device (str or torch.device): device to use for the graph parameters.
    """

    _MASK_VALUE = -9e15

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        extra_dim: Optional[Union[int, Tuple[int]]] = None,
        include_input: bool = False,
        init_param: Union[float, torch.Tensor, np.ndarray] = 1e-6,
        requires_grad: bool = False,
        device: Union[str, torch.device] = "cpu",
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            extra_dim=extra_dim,
            include_input=include_input,
            init_param=init_param,
            requires_grad=requires_grad,
            device=device,
            *args,
            **kwargs
        )

    def get_adj_matrix(self, *args, **kwargs) -> torch.Tensor:
        return torch.sigmoid(self.graph)

    def get_binary_adj_matrix(self, threshold: float = 0.5, *args, **kwargs) -> torch.Tensor:
        assert 0 <= threshold <= 1, "threshold of bernoulli graph should be in [0, 1]"

        return super().get_binary_adj_matrix(threshold, *args, **kwargs)

    def sample(
        self,
        prob_matrix: Optional[torch.Tensor],
        sample_size: Union[Tuple[int], int],
        reparameterization: Optional[str] = None,
        *args,
        **kwargs
    ):
        """sample from given or current graph probability (Bernoulli distribution).

        Args:
            graph (tensor), graph probability, use current graph parameter when given `None`.
            sample_size (tuple(int) or int), extra size of sampled graphs.

        Return:
            (tensor): [*sample_size, *extra_dim, in_dim, out_dim] shaped multiple graphs.
        """
        if prob_matrix is None:
            prob_matrix = self.get_adj_matrix()

        if isinstance(sample_size, int):
            sample_size = (sample_size,)

        sample_prob = prob_matrix[None].expand(*sample_size, *((-1,) * len(prob_matrix.shape)))

        if reparameterization is None:
            return torch.bernoulli(sample_prob)
        elif reparameterization == "gumbel-softmax":
            return F.gumbel_softmax(torch.stack((sample_prob, 1 - sample_prob)), hard=True, dim=0)[0]
        else:
            raise NotImplementedError
