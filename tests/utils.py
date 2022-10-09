from typing import cast

from stable_baselines3 import SAC
from omegaconf import DictConfig
import gym
import emei

from cmrl.util.env import make_env
from cmrl.util.creator import create_dynamics, create_replay_buffer
from cmrl.sb3_extension.online_mb_callback import OnlineModelBasedCallback
from cmrl.models.dynamics import ConstraintBasedDynamics, PlainEnsembleDynamics
from cmrl.models.transition.one_step.plain_ensemble import PlainEnsembleGaussianTransition
from cmrl.models.fake_env import VecFakeEnv

cfg = DictConfig(
    {
        "seed": 1,
        "device": "cpu",
        "task": {
            "env": "emei___BoundaryInvertedPendulumSwingUp-v0"
            "___freq_rate=${task.freq_rate}&time_step=${task.time_step}___"
            "${task.dataset}",
            "dataset": "SAC-expert-replay",
            "freq_rate": 1,
            "time_step": 0.02,
            "num_steps": 100,
            "use_ratio": 0.01,
        },
    }
)
