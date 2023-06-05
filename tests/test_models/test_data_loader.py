import gym
import emei
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader

from cmrl.models.data_loader import EnsembleBufferDataset, EnsembleBufferDataset
from cmrl.utils.env import load_offline_data


def test_buffer_dataset():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.1)

    # test for transition
    dataset = EnsembleBufferDataset(real_replay_buffer, env.observation_space, env.action_space, mech="transition")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "act_0"]
        assert list(outputs.keys()) == ["next_obs_0", "next_obs_1", "next_obs_2", "next_obs_3", "next_obs_4"]
        for key in inputs:
            assert inputs[key].shape == (128, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 1)

    # test for reward
    dataset = EnsembleBufferDataset(real_replay_buffer, env.observation_space, env.action_space, mech="reward_mech")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == [
            "obs_0",
            "obs_1",
            "obs_2",
            "obs_3",
            "obs_4",
            "act_0",
            "next_obs_0",
            "next_obs_1",
            "next_obs_2",
            "next_obs_3",
            "next_obs_4",
        ]
        assert list(outputs.keys()) == ["reward"]
        for key in inputs:
            assert inputs[key].shape == (128, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 1)

    # test for termination
    dataset = EnsembleBufferDataset(real_replay_buffer, env.observation_space, env.action_space, mech="termination_mech")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == [
            "obs_0",
            "obs_1",
            "obs_2",
            "obs_3",
            "obs_4",
            "act_0",
            "next_obs_0",
            "next_obs_1",
            "next_obs_2",
            "next_obs_3",
            "next_obs_4",
        ]
        assert list(outputs.keys()) == ["terminal"]
        for key in inputs:
            assert inputs[key].shape == (128, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 1)


def test_ensemble_buffer_dataset():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)
    # assert isinstance(env, emei.EmeiEnv)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.1)

    # test for transition
    dataset = EnsembleBufferDataset(real_replay_buffer, env.observation_space, env.action_space, mech="transition")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "act_0"]
        assert list(outputs.keys()) == ["next_obs_0", "next_obs_1", "next_obs_2", "next_obs_3", "next_obs_4"]
        for key in inputs:
            assert inputs[key].shape == (128, 7, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 7, 1)

    # test for reward
    dataset = EnsembleBufferDataset(real_replay_buffer, env.observation_space, env.action_space, mech="reward_mech")
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == [
            "obs_0",
            "obs_1",
            "obs_2",
            "obs_3",
            "obs_4",
            "act_0",
            "next_obs_0",
            "next_obs_1",
            "next_obs_2",
            "next_obs_3",
            "next_obs_4",
        ]
        assert list(outputs.keys()) == ["reward"]
        for key in inputs:
            assert inputs[key].shape == (128, 7, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 7, 1)

    # test for termination
    dataset = EnsembleBufferDataset(real_replay_buffer, env.observation_space, env.action_space, mech="termination_mech")
    loader = DataLoader(dataset, batch_size=128, drop_last=True)

    for inputs, outputs in loader:
        assert list(inputs.keys()) == [
            "obs_0",
            "obs_1",
            "obs_2",
            "obs_3",
            "obs_4",
            "act_0",
            "next_obs_0",
            "next_obs_1",
            "next_obs_2",
            "next_obs_3",
            "next_obs_4",
        ]
        assert list(outputs.keys()) == ["terminal"]
        for key in inputs:
            assert inputs[key].shape == (128, 7, 1)
        for key in outputs:
            assert outputs[key].shape == (128, 7, 1)


def test_train_valid():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.1)

    train_dataset = EnsembleBufferDataset(
        real_replay_buffer, env.observation_space, env.action_space, mech="transition", training=False
    )
    valid_dataset = EnsembleBufferDataset(
        real_replay_buffer, env.observation_space, env.action_space, mech="transition", training=True
    )

    buffer_size = real_replay_buffer.buffer_size if real_replay_buffer.full else real_replay_buffer.pos
    assert len(set(train_dataset.indexes).intersection(set(valid_dataset.indexes))) == 0
    assert len(train_dataset.indexes) + len(valid_dataset.indexes) == buffer_size


def test_ensemble_train_valid():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.1)

    ensemble_num = 7
    train_dataset = EnsembleBufferDataset(
        real_replay_buffer,
        env.observation_space,
        env.action_space,
        mech="transition",
        training=False,
        ensemble_num=ensemble_num,
        train_ensemble=False,
    )
    valid_dataset = EnsembleBufferDataset(
        real_replay_buffer,
        env.observation_space,
        env.action_space,
        mech="transition",
        training=True,
        ensemble_num=ensemble_num,
        train_ensemble=False,
    )

    buffer_size = real_replay_buffer.buffer_size if real_replay_buffer.full else real_replay_buffer.pos
    for i in range(ensemble_num):
        assert len(set(train_dataset.indexes[:, i]).intersection(set(valid_dataset.indexes[:, i]))) == 0
        assert len(train_dataset.indexes[:, i]) + len(valid_dataset.indexes[:, i]) == buffer_size


def test_mixed():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=1, time_step=0.02)
    # assert isinstance(env, emei.EmeiEnv)

    real_replay_buffer = ReplayBuffer(
        int(1e5), env.observation_space, env.action_space, "cpu", handle_timeout_termination=False
    )
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.1)

    ensemble_num = 7
    train_dataset = EnsembleBufferDataset(
        real_replay_buffer,
        env.observation_space,
        env.action_space,
        mech="transition",
        training=False,
        ensemble_num=ensemble_num,
        train_ensemble=True,
    )
    valid_dataset = EnsembleBufferDataset(
        real_replay_buffer, env.observation_space, env.action_space, mech="transition", training=True
    )

    buffer_size = real_replay_buffer.buffer_size if real_replay_buffer.full else real_replay_buffer.pos
    for i in range(ensemble_num):
        assert len(set(train_dataset.indexes[:, i]).intersection(set(valid_dataset.indexes))) == 0
        assert len(train_dataset.indexes[:, i]) + len(valid_dataset.indexes) == buffer_size
