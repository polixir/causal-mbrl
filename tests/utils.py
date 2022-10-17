from typing import cast

from stable_baselines3 import SAC
from omegaconf import DictConfig
import gym
import emei

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
