from typing import Union

import torch

from cmrl.models.networks.mlp import EnsembleMLP


class BaseRewardMech(EnsembleMLP):
    _MODEL_FILENAME = "base_reward_mech.pth"

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        deterministic: bool = False,
        ensemble_num: int = 7,
        elite_num: int = 5,
        device: Union[str, torch.device] = "cpu",
    ):
        super(BaseRewardMech, self).__init__(ensemble_num=ensemble_num, elite_num=elite_num, device=device)
        self.obs_size = obs_size
        self.action_size = action_size
        self.deterministic = deterministic

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        pass
