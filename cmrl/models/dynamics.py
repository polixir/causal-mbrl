import abc
import collections
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.models.reward_mech.base_reward_mech import BaseRewardMech
from cmrl.models.termination_mech.base_termination_mech import BaseTerminationMech
from cmrl.models.transition.base_transition import BaseTransition
from cmrl.types import InteractionBatch
from cmrl.util.transition_iterator import BootstrapIterator, TransitionIterator
from cmrl.models.causal_mech.base_causal_mech import BaseCausalMech


def split_dict(old_dict: Dict, need_keys: List[str]):
    return dict([(key, old_dict[key]) for key in need_keys])


class Dynamics:
    def __init__(
        self, transition: BaseCausalMech, reward_mech: Optional[BaseCausalMech], terminal_mech: Optional[BaseCausalMech]
    ):
        pass
