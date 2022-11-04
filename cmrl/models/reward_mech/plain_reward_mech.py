from typing import Dict, Optional, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

from cmrl.models.layers import truncated_normal_init
from cmrl.models.reward_mech.base_reward_mech import BaseRewardMech


class PlainRewardMech(BaseRewardMech):
    _MODEL_FILENAME = "plain_reward_mech.pth"

    def __init__(
        self,
        # transition info
        obs_size: int,
        action_size: int,
        deterministic: bool = False,
        # algorithm parameters
        ensemble_num: int = 7,
        elite_num: int = 5,
        learn_logvar_bounds: bool = False,
        # network parameters
        num_layers: int = 4,
        hid_size: int = 200,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        # others
        device: Union[str, torch.device] = "cpu",
    ):
        super(PlainRewardMech, self).__init__(
            obs_size=obs_size,
            action_size=action_size,
            deterministic=deterministic,
            ensemble_num=ensemble_num,
            elite_num=elite_num,
            device=device,
        )
        self.num_layers = num_layers
        self.hid_size = hid_size

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
            self.mean_and_logvar = self.create_linear_layer(hid_size, 1)
        else:
            self.mean_and_logvar = self.create_linear_layer(hid_size, 2)
            self.min_logvar = nn.Parameter(-10 * torch.ones(1), requires_grad=learn_logvar_bounds)
            self.max_logvar = nn.Parameter(0.5 * torch.ones(1), requires_grad=learn_logvar_bounds)

        self.apply(truncated_normal_init)
        self.to(self.device)

    def forward(
        self,
        batch_obs: torch.Tensor,  # shape: ensemble_num, batch_size, state_size
        batch_action: torch.Tensor,  # shape: ensemble_num, batch_size, action_size
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(batch_obs.shape) == 3 and batch_obs.shape[-1] == self.obs_size
        assert len(batch_action.shape) == 3 and batch_action.shape[-1] == self.action_size

        hidden = self.hidden_layers(torch.concat([batch_obs, batch_action], dim=-1))
        mean_and_logvar = self.mean_and_logvar(hidden)

        if self.deterministic:
            mean, logvar = mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., :1]
            logvar = mean_and_logvar[..., 1:]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar
