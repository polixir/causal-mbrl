import gym
import emei
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader

from cmrl.models.data_loader import OfflineDataset, EnsembleOfflineDataset
from cmrl.algorithms.util import load_offline_data


def test_offline_dataset():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)
    # assert isinstance(env, emei.EmeiEnv)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.1)

    # test for transition
    dataset = OfflineDataset(real_replay_buffer, env.observation_space, env.action_space, mech="transition")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "act_0"]
        assert list(outputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4"]
        for key in inputs:
            assert inputs[key].shape == (128, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 1)

    # test for reward
    dataset = OfflineDataset(real_replay_buffer, env.observation_space, env.action_space, mech="reward_mech")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "act_0"]
        assert list(outputs.keys()) == ["reward"]
        for key in inputs:
            assert inputs[key].shape == (128, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 1)

    # test for termination
    dataset = OfflineDataset(real_replay_buffer, env.observation_space, env.action_space, mech="termination_mech")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "act_0"]
        assert list(outputs.keys()) == ["terminal"]
        for key in inputs:
            assert inputs[key].shape == (128, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 1)


def test_ensemble_offline_dataset():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)
    # assert isinstance(env, emei.EmeiEnv)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.1)

    # test for transition
    dataset = EnsembleOfflineDataset(real_replay_buffer, env.observation_space, env.action_space, mech="transition")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "act_0"]
        assert list(outputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4"]
        for key in inputs:
            assert inputs[key].shape == (128, 7, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 7, 1)

    # test for reward
    dataset = EnsembleOfflineDataset(real_replay_buffer, env.observation_space, env.action_space, mech="reward_mech")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "act_0"]
        assert list(outputs.keys()) == ["reward"]
        for key in inputs:
            assert inputs[key].shape == (128, 7, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 7, 1)

    # test for termination
    dataset = EnsembleOfflineDataset(real_replay_buffer, env.observation_space, env.action_space, mech="termination_mech")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "act_0"]
        assert list(outputs.keys()) == ["terminal"]
        for key in inputs:
            assert inputs[key].shape == (128, 7, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 7, 1)
