from typing import Optional

from omegaconf import DictConfig

from cmrl.models.fake_env import VecFakeEnv
from cmrl.algorithms.base_algorithm import BaseAlgorithm


class MOPO(BaseAlgorithm):
    def __init__(
        self,
        cfg: DictConfig,
        work_dir: Optional[str] = None,
    ):
        super(MOPO, self).__init__(cfg, work_dir)

    def get_fake_env(self) -> VecFakeEnv:
        return self.partial_fake_env(
            deterministic=self.cfg.algorithm.deterministic,
            max_episode_steps=self.cfg.algorithm.branch_rollout_length,
            branch_rollout=True,
        )
