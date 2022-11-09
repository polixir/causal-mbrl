from typing import Optional

from omegaconf import DictConfig

from cmrl.models.fake_env import VecFakeEnv
from cmrl.algorithms.base_algorithm import BaseAlgorithm
from cmrl.utils.env import load_offline_data


class MOPO(BaseAlgorithm):
    def __init__(
        self,
        cfg: DictConfig,
        work_dir: Optional[str] = None,
    ):
        super(MOPO, self).__init__(cfg, work_dir)

    @property
    def fake_env(self) -> VecFakeEnv:
        return self.partial_fake_env(
            deterministic=self.cfg.algorithm.deterministic,
            max_episode_steps=self.cfg.algorithm.branch_rollout_length,
            branch_rollout=True,
        )

    def _setup_learn(self):
        load_offline_data(self.env, self.real_replay_buffer, self.cfg.task.dataset, self.cfg.task.use_ratio)

        self.dynamics.learn(
            real_replay_buffer=self.real_replay_buffer,
            longest_epoch=self.cfg.task.longest_epoch,
            improvement_threshold=self.cfg.task.improvement_threshold,
            patience=self.cfg.task.patience,
            work_dir=self.work_dir,
        )
