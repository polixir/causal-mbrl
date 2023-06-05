import abc
import pathlib
from typing import Optional, Tuple, Union

import torch


class BaseGraph(abc.ABC):
    """Base abstract class for all graph models.

    All classes derived from `BaseGraph` must implement the following methods:

        - ``parameters``: the graph parameters property.
        - ``get_adj_matrix``: get the (raw) adjacency matrix.
        - ``get_binary_adj_matrix``: get the binary format of the adjacency matrix.
        - ``save``: save the graph data
        - ``load``: load the graph data

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        extra_dim (int | tuple(int) | None): extra dimensions (multi-graph).
        include_input (bool): whether include input variables in the output variables.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        extra_dim: Optional[Union[int, Tuple[int]]] = None,
        include_input: bool = False,
        *args,
        **kwargs
    ) -> None:
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._extra_dim = extra_dim
        self._include_input = include_input

        assert not (include_input and out_dim < in_dim), "Once include input, the out dimension must >= in dimension"

    @property
    @abc.abstractmethod
    def parameters(self) -> Tuple[torch.Tensor]:
        """Get the graph parameters (raw graph).

        Returns: (tuple of tensor) the true graph parameters
        """

    @abc.abstractmethod
    def get_adj_matrix(self, *args, **kwargs) -> torch.Tensor:
        """Get the raw adjacency matrix.

        Returns:
            (tensor): the raw adjacency matrix tensor, shape [in_dim, out_dim];
        """

    @abc.abstractmethod
    def get_binary_adj_matrix(self, *args, **kwargs) -> torch.Tensor:
        """Get the binary adjacency matrix.

        Returns:
            (tensor): the binary adjacency matrix tensor, shape [in_dim, out_dim];
            graph[i, j] == 1 represents i causes j
        """

    @abc.abstractmethod
    def save(self, save_dir: Union[str, pathlib.Path]):
        """Save the model to the given directory."""

    @abc.abstractmethod
    def load(self, load_dir: Union[str, pathlib.Path]):
        """Load the model from the given path."""
