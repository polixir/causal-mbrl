import gym
import emei
from stable_baselines3.common.buffers import ReplayBuffer

from cmrl.models.data_loader import OfflineDataset
from cmrl.algorithms.util import load_offline_data


def test_offline_dataset():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)
    # assert isinstance(env, emei.EmeiEnv)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay")

    OfflineDataset(real_replay_buffer, env.observation_space, env.action_space, mech="transition")
