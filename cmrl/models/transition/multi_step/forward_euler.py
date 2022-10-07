import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

from cmrl.models.layers import EnsembleLinearLayer, truncated_normal_init
from cmrl.models.transition.base_transition import BaseEnsembleTransition


class ForwardEulerTransition(BaseEnsembleTransition):
    def __init__(
        self, one_step_transition: BaseEnsembleTransition, repeat_times: int = 2
    ):
        super().__init__(
            obs_size=one_step_transition.obs_size,
            action_size=one_step_transition.action_size,
            ensemble_num=one_step_transition.ensemble_num,
            deterministic=one_step_transition.deterministic,
            device=one_step_transition.device,
        )

        self.one_step_transition = one_step_transition
        self.repeat_times = repeat_times

        if hasattr(self.one_step_transition, "max_logvar"):
            self.max_logvar = one_step_transition.max_logvar
            self.min_logvar = one_step_transition.min_logvar

        if hasattr(self.one_step_transition, "input_mask"):
            self.input_mask = self.one_step_transition.input_mask
            self.set_input_mask = self.one_step_transition.set_input_mask

    def set_elite_members(self, elite_indices: Sequence[int]):
        self.one_step_transition.set_elite_members(elite_indices)

    def forward(
        self,
        batch_obs: torch.Tensor,
        batch_action: torch.Tensor,
        only_elite: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logvar = torch.zeros(batch_obs.shape, device=self.device)
        mean = batch_obs
        for t in range(self.repeat_times):
            mean, logvar = self.one_step_transition.forward(
                mean, batch_action.clone(), only_elite
            )
        return mean, logvar
