import numpy as np
import torch

import cmrl.third_party.pytorch_sac as pytorch_sac

from .core import Agent


class SACAgent(Agent):
    def __init__(self, sac_agent: pytorch_sac.SAC):
        self.sac_agent = sac_agent

    def act(self, obs: np.ndarray, sample: bool = False, batched: bool = False, **kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation (or batch of observations) for which the action
                is needed.
            sample (bool): if ``True`` the agent samples actions from its policy, otherwise it
                returns the mean policy value. Defaults to ``False``.
            batched (bool): if ``True`` signals to the agent that the obs should be interpreted
                as a batch.

        Returns:
            (np.ndarray): the action.
        """
        with torch.no_grad():
            return self.sac_agent.select_action(obs, batched=batched, evaluate=not sample)
