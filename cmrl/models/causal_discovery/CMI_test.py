from typing import Dict, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

# from cmrl.models.layers import ParallelEnsembleLinearLayer, truncated_normal_init
from cmrl.models.networks.mlp import EnsembleMLP


class TransitionConditionalMutualInformationTest(EnsembleMLP):
    _MODEL_FILENAME = "conditional_mutual_information_test.pth"

    def __init__(
        self,
        # transition info
        obs_size: int,
        action_size: int,
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
            ensemble_num=ensemble_num,
            elite_num=elite_num,
            device=device,
        )
        self.obs_size = obs_size
        self.action_size = action_size
        self.residual = residual
        self.learn_logvar_bounds = learn_logvar_bounds

        self.num_layers = num_layers
        self.hid_size = hid_size

        self.parallel_num = self.obs_size + self.action_size + 1

        self._input_mask = 1 - torch.eye(self.parallel_num, self.obs_size + self.action_size).to(self.device)

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

        self.mean_and_logvar = self.create_linear_layer(hid_size, 2 * self.obs_size)
        self.min_logvar = nn.Parameter(
            -10 * torch.ones(self.parallel_num, 1, 1, self.obs_size), requires_grad=learn_logvar_bounds
        )
        self.max_logvar = nn.Parameter(
            0.5 * torch.ones(self.parallel_num, 1, 1, self.obs_size), requires_grad=learn_logvar_bounds
        )

        # self.apply(truncated_normal_init)
        self.to(self.device)

    # def create_linear_layer(self, l_in, l_out):
    #     return ParallelEnsembleLinearLayer(l_in, l_out, parallel_num=self.parallel_num, ensemble_num=self.ensemble_num)

    @property
    def input_mask(self):
        return self._input_mask

    def mask_input(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        assert self._input_mask.ndim == 2
        input_mask = self._input_mask[:, None, None, :]
        return x * input_mask

    def forward(
        self,
        batch_obs: torch.Tensor,  # shape: (parallel_num, )ensemble_num, batch_size, obs_size
        batch_action: torch.Tensor,  # shape: ensemble_num, batch_size, action_size
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(batch_action.shape) == 3 and batch_action.shape[-1] == self.action_size

        batch_action = batch_action.repeat((self.parallel_num, 1, 1, 1))
        if len(batch_obs.shape) == 3:  # non-repeat or first repeat
            batch_obs = batch_obs.repeat((self.parallel_num, 1, 1, 1))

        batch_input = torch.concat([batch_obs, batch_action], dim=-1)

        masked_input = self.mask_input(batch_input)
        hidden = self.hidden_layers(masked_input)
        mean_and_logvar = self.mean_and_logvar(hidden)

        mean = mean_and_logvar[..., : self.obs_size]
        logvar = mean_and_logvar[..., self.obs_size :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if self.residual:
            mean += batch_obs

        return mean, logvar

    def get_nll_loss(self, model_in: Dict[(str, torch.Tensor)], target: torch.Tensor) -> torch.Tensor:
        pred_mean, pred_logvar = self.forward(**model_in)
        target = target.repeat((self.parallel_num, 1, 1, 1))

        nll_loss = gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
        nll_loss += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll_loss
