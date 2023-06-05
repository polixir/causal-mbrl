import gym
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader

from cmrl.models.causal_mech.oracle_mech import OracleMech
from cmrl.models.data_loader import EnsembleBufferDataset, EnsembleBufferDataset, collate_fn
from cmrl.utils.creator import parse_space
from cmrl.utils.env import load_offline_data


def prepare(freq_rate):
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=freq_rate, time_step=0.02)

    real_replay_buffer = ReplayBuffer(1000, env.observation_space, env.action_space, "cpu", handle_timeout_termination=False)
    load_offline_data(env, real_replay_buffer, "SAC-expert-replay", use_ratio=0.001)

    ensemble_num = 7
    # test for transition
    train_dataset = EnsembleBufferDataset(
        real_replay_buffer,
        env.observation_space,
        env.action_space,
        training=False,
        mech="transition",
        train_ensemble=True,
        ensemble_num=ensemble_num,
    )
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)
    valid_dataset = EnsembleBufferDataset(
        real_replay_buffer, env.observation_space, env.action_space, training=True, mech="transition", repeat=ensemble_num
    )
    valid_loader = DataLoader(valid_dataset, batch_size=8, collate_fn=collate_fn)

    input_variables = parse_space(env.observation_space, "obs") + parse_space(env.action_space, "act")
    output_variables = parse_space(env.observation_space, "next_obs")

    return input_variables, output_variables, train_loader, valid_loader


def test_inv_pendulum_single_step():
    input_variables, output_variables, train_loader, valid_loader = prepare(freq_rate=1)

    mech = OracleMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )

    mech.learn(train_loader, valid_loader, longest_epoch=1)


def test_inv_pendulum_multi_step():
    input_variables, output_variables, train_loader, valid_loader = prepare(freq_rate=2)

    mech = OracleMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
        multi_step="forward-euler 2",
    )

    mech.learn(train_loader, valid_loader, longest_epoch=1)


if __name__ == "__main__":
    test_inv_pendulum_multi_step()
