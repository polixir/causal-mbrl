from typing import Optional

from omegaconf import DictConfig

from cmrl.algorithms.base_algorithm import BaseAlgorithm
from cmrl.utils.env import load_offline_data
from cmrl.algorithms.util import maybe_load_offline_model


class OfflineDyna(BaseAlgorithm):
    def __init__(
        self,
        cfg: DictConfig,
        work_dir: Optional[str] = None,
    ):
        super(OfflineDyna, self).__init__(cfg, work_dir)

    def _setup_learn(self):
        load_offline_data(self.env, self.real_replay_buffer, self.cfg.task.dataset, self.cfg.task.use_ratio)

        if self.cfg.task.get("auto_load_offline_model", False):
            existed_trained_model = maybe_load_offline_model(self.dynamics, self.cfg, work_dir=self.work_dir)
        else:
            existed_trained_model = None
        if not existed_trained_model:
            self.dynamics.learn(
                real_replay_buffer=self.real_replay_buffer,
                work_dir=self.work_dir,
            )
