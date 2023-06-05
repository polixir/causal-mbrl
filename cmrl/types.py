from typing import Callable, Optional, Tuple, Union

import torch

# (next_obs, pre_obs, action) -> reward
RewardFnType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
# (next_obs, pre_obs, action) -> terminal
TermFnType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
InitObsFnType = Callable[[int], torch.Tensor]
Obs2StateFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
State2ObsFnType = Callable[[torch.Tensor], torch.Tensor]
