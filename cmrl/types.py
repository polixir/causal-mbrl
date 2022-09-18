from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

# (next_obs, pre_obs, action) -> reward
RewardFnType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
# (next_obs, pre_obs, action) -> terminal
TermFnType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
InitObsFnType = Callable[[int], torch.Tensor]
ObsProcessFnType = Callable[[np.ndarray], np.ndarray]

TensorType = Union[torch.Tensor, np.ndarray]
TrajectoryEvalFnType = Callable[[TensorType, torch.Tensor], torch.Tensor]
# obs, action, next_obs, reward, done
InteractionData = Tuple[TensorType, TensorType, TensorType, TensorType, TensorType]


@dataclass
class InteractionBatch:
    """Represents a batch of transitions"""

    batch_obs: Optional[TensorType]
    batch_action: Optional[TensorType]
    batch_next_obs: Optional[TensorType]
    batch_reward: Optional[TensorType]
    batch_done: Optional[TensorType]

    @property
    def attrs(self):
        return ["batch_obs", "batch_action", "batch_next_obs", "batch_reward", "batch_done"]

    def __len__(self):
        return self.batch_obs.shape[0]

    def as_tuple(self) -> InteractionData:
        return self.batch_obs, self.batch_action, self.batch_next_obs, self.batch_reward, self.batch_done

    def __getitem__(self, item):
        return InteractionBatch(
            self.batch_obs[item],
            self.batch_action[item],
            self.batch_next_obs[item],
            self.batch_reward[item],
            self.batch_done[item],
        )

    @staticmethod
    def _get_new_shape(old_shape: Tuple[int, ...], batch_size: int):
        new_shape = list((1,) + old_shape)
        new_shape[0] = batch_size
        new_shape[1] = old_shape[0] // batch_size
        return tuple(new_shape)

    def add_new_batch_dim(self, batch_size: int):
        if not len(self) % batch_size == 0:
            raise ValueError(
                "Current batch of transitions size is not a "
                "multiple of the new batch size. "
            )
        return InteractionBatch(
            self.batch_obs.reshape(self._get_new_shape(self.batch_obs.shape, batch_size)),
            self.batch_action.reshape(self._get_new_shape(self.batch_action.shape, batch_size)),
            self.batch_next_obs.reshape(self._get_new_shape(self.batch_obs.shape, batch_size)),
            self.batch_reward.reshape(self._get_new_shape(self.batch_reward.shape, batch_size)),
            self.batch_done.reshape(self._get_new_shape(self.batch_done.shape, batch_size)),
        )


ModelInput = Union[torch.Tensor, InteractionBatch]
