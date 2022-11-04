import pathlib
from typing import List, Optional, Sequence, Union
from abc import abstractmethod

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig


class BaseNetwork(nn.Module):
    def __init__(self, **kwargs):
        """Base class of all neural network.

        Args:
            network_cfg:
        """
        super(BaseNetwork, self).__init__()

        self._model_filename = "base_network.pth"
        self._save_attrs: List[str] = ["state_dict"]
        self._layers: Optional[nn.ModuleList] = None

        self.build()

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {}
        for attr in self._save_attrs:
            if attr == "state_dict":
                model_dict["state_dict"] = self.state_dict()
            else:
                model_dict[attr] = getattr(self, attr)
        torch.save(model_dict, pathlib.Path(save_dir) / self._model_filename)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._model_filename, map_location=self.device)
        for attr in model_dict:
            if attr == "state_dict":
                self.load_state_dict(model_dict["state_dict"])
            else:
                getattr(self, attr)(model_dict[attr])

    def forward(self, x) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @property
    def save_attrs(self):
        return self._save_attrs

    @property
    def model_filename(self):
        return self._model_filename

    @property
    def device(self):
        return next(iter(self.parameters())).device


def create_activation(activation_fn_cfg: DictConfig):
    if activation_fn_cfg is None:
        return nn.ReLU()
    else:
        return hydra.utils.instantiate(activation_fn_cfg)
