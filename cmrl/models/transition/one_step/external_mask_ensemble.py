import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

import cmrl.types
from cmrl.models.util import to_tensor
from cmrl.models.layers import ParallelEnsembleLinearLayer, truncated_normal_init
from cmrl.models.transition.base_transition import BaseEnsembleTransition


class ExternalMaskEnsembleGaussianTransition(BaseEnsembleTransition):
    """Implements an ensemble of multi-layer perceptrons each modeling a Gaussian distribution
        corresponding to each independent dimension.

    Args:
        obs_size (int): size of state.
        action_size (int): size of action.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_num (int): the number of members in the ensemble. Defaults to 1.
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        deterministic (bool): if ``True``, the model predicts the mean and logvar of the conditional
            gaussian distribution, otherwise only predicts the mean. Defaults to ``False``.
        residual (bool): if ``True``, the model predicts the residual of output and input. Defaults to ``True``.
        learn_logvar_bounds (bool): if ``True``, the log-var bounds will be learned, otherwise
            they will be constant. Defaults to ``False``.
        activation_fn_cfg (dict or omegaconf.DictConfig, optional): configuration of the
            desired activation function. Defaults to torch.nn.ReLU when ``None``.
    """

    _MODEL_FILENAME = "external_mask_ensemble_transition.pth"

    def __init__(
            self,
            # transition info
            obs_size: int,
            action_size: int,
            deterministic: bool = False,
            # algorithm parameters
            ensemble_num: int = 7,
            elite_num: int = 5,
            residual: bool = True,
            learn_logvar_bounds: bool = False,
            # network parameters
            num_layers: int = 4,
            hid_size: int = 200,
            activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
            # others
            device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(obs_size=obs_size,
                         action_size=action_size,
                         deterministic=deterministic,
                         ensemble_num=ensemble_num,
                         elite_num=elite_num,
                         device=device)
        self.residual = residual
        self.learn_logvar_bounds = learn_logvar_bounds

        self.num_layers = num_layers
        self.hid_size = hid_size

        def create_activation():
            if activation_fn_cfg is None:
                return nn.ReLU()
            else:
                return hydra.utils.instantiate(activation_fn_cfg)

        hidden_layers = [
            nn.Sequential(self.create_linear_layer(obs_size + action_size, hid_size), create_activation())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    self.create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)

        if deterministic:
            self.mean_and_logvar = self.create_linear_layer(hid_size, 1)
        else:
            self.mean_and_logvar = self.create_linear_layer(hid_size, 2)
            self.min_logvar = nn.Parameter(
                -10 * torch.ones(obs_size, 1, 1, 1), requires_grad=learn_logvar_bounds
            )
            self.max_logvar = nn.Parameter(
                0.5 * torch.ones(obs_size, 1, 1, 1), requires_grad=learn_logvar_bounds
            )

        self.apply(truncated_normal_init)
        self.to(self.device)

    def create_linear_layer(self, l_in, l_out):
        return ParallelEnsembleLinearLayer(l_in, l_out,
                                           parallel_num=self.obs_size,
                                           ensemble_num=self.ensemble_num)

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_members is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_members)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_members)
            self.mean_and_logvar.toggle_use_only_elite()

    def set_input_mask(self, mask: cmrl.types.TensorType):
        self.input_mask = to_tensor(mask).to(self.device)

    def mask_input(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        assert self.input_mask is not None
        assert 2 <= self.input_mask.ndim <= 4
        assert x.shape[0] == self.input_mask.shape[0] and x.shape[-1] == self.input_mask.shape[-1]

        if self.input_mask.ndim == 2:
            # [parallel_size x in_dim]
            input_mask = self.input_mask[:, None, None, :]
        elif self.input_mask.ndim == 3:
            if self.input_mask.shape[1] == x.shape[1]:
                # [parallel_size x ensemble_size x in_dim]
                input_mask = self.input_mask[:, :, None, :]
            elif self.input_mask.shape[1] == x.shape[2]:
                # [parallel_size x batch_size x in_dim]
                input_mask = self.input_mask[:, None, :, :]
            else:
                raise RuntimeError(
                    "input mask shape %a does not match x shape %a" % (self.input_mask.shape, x.shape)
                )
        else:
            assert self.input_mask.shape == x.shape
            input_mask = self.input_mask

        x *= input_mask
        return x

    def forward(
            self,
            batch_obs: torch.Tensor,  # shape: ensemble_num, batch_size, obs_size
            batch_action: torch.Tensor,  # shape: ensemble_num, batch_size, action_size
            only_elite: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(batch_obs.shape) == 3 and batch_obs.shape[-1] == self.obs_size
        assert len(batch_action.shape) == 3 and batch_action.shape[-1] == self.action_size

        self._maybe_toggle_layers_use_only_elite(only_elite)
        repeated_input = torch.concat([batch_obs, batch_action], dim=-1).repeat((self.obs_size, 1, 1, 1))
        masked_input = self.mask_input(repeated_input)
        hidden = self.hidden_layers(masked_input)
        mean_and_logvar = self.mean_and_logvar(hidden)
        self._maybe_toggle_layers_use_only_elite(only_elite)

        if self.deterministic:
            mean, logvar = mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., : 1]
            logvar = mean_and_logvar[..., 1:]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        mean = torch.transpose(mean, 0, -1)[0]
        if logvar is not None:
            logvar = torch.transpose(logvar, 0, -1)[0]

        if self.residual:
            mean += batch_obs.detach()

        return mean, logvar

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_members,
            "input_mask": self.input_mask,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FILENAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FILENAME)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_members = model_dict["elite_models"]
        self.input_mask = model_dict["input_mask"]
