import abc
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class BaseGraph(nn.Module, abc.ABC):
    """Base abstract class for all graph models.

    All classes derived from `BaseGraph` must implement the following methods:

        - ``forward``: computes the graph (parameters).
        - ``update``: updates the structural parameters.
        - ``get_binary_graph``: gets the binary graph.

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        device (str or torch.device): device to use for the structural parameters.
    """

    _GRAPH_FNAME = "graph.pth"

    def __init__(self, in_dim: int, out_dim: int, device: Union[str, torch.device] = "cpu", *args, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Computes the graph parameters.

        Returns:
            (tuple of tensors): all tensors representing the output
                graph (e.g. existence and orientation)
        """

    @abc.abstractmethod
    def get_binary_graph(self, *args, **kwargs) -> torch.Tensor:
        """Gets the binary graph.

        Returns:
            (tensor): the binary graph tensor, shape [in_dim, out_dim];
            graph[i, j] == 1 represents i causes j
        """

    def get_mask(self, *args, **kwargs) -> torch.Tensor:
        # [..., in_dim, out_dim]
        binary_mat = self.get_binary_graph(*args, **kwargs)
        # [..., out_dim, in_dim], mask apply on the input for each output variable
        return binary_mat.transpose(-1, -2)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        torch.save(self.state_dict(), pathlib.Path(save_dir) / self._GRAPH_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        self.load_state_dict(torch.load(pathlib.Path(load_dir) / self._GRAPH_FNAME, map_location=self.device))


class BaseEnsembleGraph(BaseGraph, abc.ABC):
    """Base abstract class for all ensemble of bootstrapped 1-D graph models.

    Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`.
            - "expectation": the output for each element in the batch will be the mean across
              models.
            - "majority": the output for each element in the batch will be determined by the
              majority voting with the models (only for binary edge).

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outpus of all models.

    Args:
        num_members (int): number of models in the ensemble.
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        device (str or torch.device): device to use for the model.
        propagation_method (str, optional): the uncertainty method to use. Defaults to ``None``.
    """

    def __init__(
        self,
        num_members: int,
        in_dim: int,
        out_dim: int,
        device: Union[str, torch.device],
        propagation_method: str,
        *args,
        **kwargs
    ):
        super().__init__(in_dim, out_dim, device, *args, **kwargs)
        self.num_members = num_members
        self.propagation_method = propagation_method
        self.device = torch.device(device)

    def __len__(self):
        return self.num_members

    def set_elite(self, elite_grpahs: Sequence[int]):
        """For ensemble graphs, indicates if some graphs should be considered elite."""
        pass

    @abc.abstractmethod
    def sample_propagation_indices(self, batch_size: int, rng: torch.Generator) -> torch.Tensor:
        """Samples uncertainty propagation indices.

        Args:
            batch_size (int): the desired batch size.
            rng (`torch.Generator`): a random number generator to use for sampling.
        Returns:
            (tensor) with ``batch_size`` integers from [0, ``self.num_members``).
        """

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        self.propagation_method = propagation_method
