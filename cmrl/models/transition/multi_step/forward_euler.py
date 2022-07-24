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
    def __init__(self,
                 one_step_transition,
                 repeat_times: int = 3
                 ):
        super().__init__(obs_size=one_step_transition.obs_size,
                         action_size=one_step_transition.action_size,
                         ensemble_num=one_step_transition.ensemble_num,
                         deterministic=one_step_transition.deterministic,
                         device=one_step_transition.device)

        self.one_step_transition = one_step_transition
        self.repeat_times = repeat_times

    def save(self, save_dir: Union[str, pathlib.Path]):
        pass

    def load(self, save_dir: Union[str, pathlib.Path]):
        pass

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.ensemble_num:
            self.one_step_transition.elite_members = list(elite_indices)

    def forward(
            self,
            batch_obs: torch.Tensor,
            batch_action: torch.Tensor,
            only_elite: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        total_logvar = torch.zeros(batch_obs.shape)
        mean = batch_obs
        for t in range(self.repeat_times):
            mean, logvar = self.one_step_transition.forward(mean, batch_action, only_elite)
            if not self.one_step_transition.deterministic:
                total_logvar += logvar
        return mean, total_logvar
