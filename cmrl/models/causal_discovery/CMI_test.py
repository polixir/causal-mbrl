from typing import Dict, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

import cmrl.types
from cmrl.models.layers import ParallelEnsembleLinearLayer, truncated_normal_init
from cmrl.models.transition.base_transition import BaseEnsembleTransition
from cmrl.models.util import to_tensor


class ConditionalMutualInformationTest(BaseEnsembleTransition):
    _MODEL_FILENAME = "miss_one_mask_ensemble_transition.pth"

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
        super().__init__(
            obs_size=obs_size,
            action_size=action_size,
            deterministic=deterministic,
            ensemble_num=ensemble_num,
            elite_num=elite_num,
            device=device,
        )
        self.residual = residual
        self.learn_logvar_bounds = learn_logvar_bounds

        self.num_layers = num_layers
        self.hid_size = hid_size

        self._input_mask: Optional[torch.Tensor] = torch.ones((obs_size, obs_size + action_size)).to(device)
        self.add_save_attr("input_mask")

        def create_activation():
            if activation_fn_cfg is None:
                return nn.ReLU()
            else:
                return hydra.utils.instantiate(activation_fn_cfg)

        hidden_layers = [
            nn.Sequential(
                self.create_linear_layer(obs_size + action_size, hid_size),
                create_activation(),
            )
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
            self.mean_and_logvar = self.create_linear_layer(hid_size, self.obs_size)
        else:
            self.mean_and_logvar = self.create_linear_layer(hid_size, 2 * self.obs_size)
            self.min_logvar = nn.Parameter(-10 * torch.ones(obs_size, 1, 1, self.obs_size),
                                           requires_grad=learn_logvar_bounds)
            self.max_logvar = nn.Parameter(0.5 * torch.ones(obs_size, 1, 1, self.obs_size),
                                           requires_grad=learn_logvar_bounds)

        self.apply(truncated_normal_init)
        self.to(self.device)

    def create_linear_layer(self, l_in, l_out):
        return ParallelEnsembleLinearLayer(l_in, l_out, parallel_num=self.obs_size + self.action_size,
                                           ensemble_num=self.ensemble_num)

    def set_input_mask(self, mask: cmrl.types.TensorType):
        self._input_mask = to_tensor(mask).to(self.device)

    @property
    def input_mask(self):
        return self._input_mask

    def mask_input(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        assert self._input_mask is not None
        assert 2 <= self._input_mask.ndim <= 4
        assert x.shape[0] == self._input_mask.shape[0] and x.shape[-1] == self._input_mask.shape[-1]

        if self._input_mask.ndim == 2:
            # [parallel_size x in_dim]
            input_mask = self._input_mask[:, None, None, :]
        elif self._input_mask.ndim == 3:
            if self._input_mask.shape[1] == x.shape[1]:
                # [parallel_size x ensemble_size x in_dim]
                input_mask = self._input_mask[:, :, None, :]
            elif self._input_mask.shape[1] == x.shape[2]:
                # [parallel_size x batch_size x in_dim]
                input_mask = self._input_mask[:, None, :, :]
            else:
                raise RuntimeError("input mask shape %a does not match x shape %a" % (self._input_mask.shape, x.shape))
        else:
            assert self._input_mask.shape == x.shape
            input_mask = self._input_mask

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

        repeated_input = torch.concat([batch_obs, batch_action], dim=-1).repeat((self.obs_size, 1, 1, 1))
        masked_input = self.mask_input(repeated_input)
        hidden = self.hidden_layers(masked_input)
        mean_and_logvar = self.mean_and_logvar(hidden)

        if self.deterministic:
            mean, logvar = mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., :1]
            logvar = mean_and_logvar[..., 1:]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        mean = torch.transpose(mean, 0, -1)[0]
        if logvar is not None:
            logvar = torch.transpose(logvar, 0, -1)[0]

        if self.residual:
            mean += batch_obs.detach()

        return mean, logvar
