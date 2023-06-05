from typing import Optional

from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from cmrl.models.fake_env import VecFakeEnv
from cmrl.algorithms.base_algorithm import BaseAlgorithm
from cmrl.sb3_extension.online_mb_callback import OnlineModelBasedCallback


class MBPO(BaseAlgorithm):
    def __init__(
        self,
        cfg: DictConfig,
        work_dir: Optional[str] = None,
    ):
        super(MBPO, self).__init__(cfg, work_dir)

    @property
    def fake_env(self) -> VecFakeEnv:
        return self.partial_fake_env(
            deterministic=self.cfg.algorithm.deterministic,
            max_episode_steps=self.cfg.algorithm.branch_rollout_length,
            branch_rollout=True,
        )

    @property
    def callback(self) -> BaseCallback:
        eval_callback = super(MBPO, self).callback
        omb_callback = OnlineModelBasedCallback(
            self.env,
            self.dynamics,
            self.real_replay_buffer,
            total_online_timesteps=self.cfg.task.online_num_steps,
            initial_exploration_steps=self.cfg.algorithm.initial_exploration_steps,
            freq_train_model=self.cfg.task.freq_train_model,
            device=self.cfg.device,
        )

        return CallbackList([eval_callback, omb_callback])
