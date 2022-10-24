import pathlib
from typing import List, Optional, Sequence, Union
from abc import abstractmethod

import torch
import torch.nn as nn
from omegaconf import DictConfig

from cmrl.models.util import gaussian_nll
from cmrl.models.layers import ParallelLinear


class BaseNetwork(nn.Module):
    _MODEL_FILENAME = "base_network.pth"

    def __init__(self, network_cfg: DictConfig):
        super(BaseNetwork, self).__init__()

        self.network_cfg = network_cfg

        self._model_save_attrs: List[str] = ["state_dict"]
        self._layers: Optional[nn.Module] = None

        self.build_network()

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {}
        for attr in self._model_save_attrs:
            if attr == "state_dict":
                model_dict["state_dict"] = self.state_dict()
            else:
                model_dict[attr] = getattr(self, attr)
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FILENAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FILENAME, map_location=self.device)
        for attr in model_dict:
            if attr == "state_dict":
                self.load_state_dict(model_dict["state_dict"])
            else:
                getattr(self, "set_" + attr)(model_dict[attr])

    @abstractmethod
    def build_network(self):
        raise NotImplementedError

    @property
    def save_attrs(self):
        return self._model_save_attrs

    @property
    def model_file_name(self):
        return self._MODEL_FILENAME

    @property
    def device(self):
        return next(iter(self.parameters())).device
