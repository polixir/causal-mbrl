from typing import Union

import torch
from gym import Space

from cmrl.models.base_cuasal_mech import BaseCausalMechanism


class BaseTransition(BaseCausalMechanism):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        deterministic: bool,
    ):
        super(BaseTransition, self).__init__(obs_space, action_space, deterministic)
