import gym
from stable_baselines3.common.buffers import ReplayBuffer
import torch
from torch.utils.data import DataLoader

from cmrl.models.causal_mech.CMI_test import CMITestMech
from cmrl.models.data_loader import EnsembleBufferDataset, EnsembleBufferDataset, collate_fn
from cmrl.utils.creator import parse_space
from cmrl.utils.env import load_offline_data
from cmrl.models.causal_mech.util import variable_loss_func


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


def test_mask():
    input_variables, output_variables, train_loader, valid_loader = prepare(freq_rate=1)

    mech = CMITestMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )

    for inputs, targets in train_loader:
        batch_size, extra_dim = mech.get_inputs_batch_size(inputs)

        inputs_tensor = torch.zeros(mech.ensemble_num, batch_size, mech.input_var_num, mech.encoder_output_dim).to(mech.device)
        for i, var in enumerate(mech.input_variables):
            out = mech.variable_encoders[var.name](inputs[var.name].to(mech.device))
            inputs_tensor[:, :, i] = out

        mask = None
        masked_inputs_tensor = mech.reduce_encoder_output(inputs_tensor, mask)
        assert masked_inputs_tensor.shape == (mech.output_var_num, mech.ensemble_num, batch_size, mech.encoder_output_dim)

        mask = torch.ones(mech.ensemble_num, batch_size, mech.input_var_num).to(mech.device)
        masked_inputs_tensor = mech.reduce_encoder_output(inputs_tensor, mask)
        assert masked_inputs_tensor.shape == (mech.ensemble_num, batch_size, mech.encoder_output_dim)

        mask = torch.ones(mech.output_var_num, mech.ensemble_num, batch_size, mech.input_var_num).to(mech.device)
        masked_inputs_tensor = mech.reduce_encoder_output(inputs_tensor, mask)
        assert masked_inputs_tensor.shape == (mech.output_var_num, mech.ensemble_num, batch_size, mech.encoder_output_dim)

        mask = torch.ones(mech.input_var_num + 1, mech.output_var_num, mech.ensemble_num, batch_size, mech.input_var_num).to(
            mech.device
        )
        masked_inputs_tensor = mech.reduce_encoder_output(inputs_tensor, mask)
        assert masked_inputs_tensor.shape == (
            mech.input_var_num + 1,
            mech.output_var_num,
            mech.ensemble_num,
            batch_size,
            mech.encoder_output_dim,
        )

        break


def test_CMI_forward():
    input_variables, output_variables, train_loader, valid_loader = prepare(freq_rate=1)

    mech = CMITestMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )

    for inputs, targets in train_loader:
        outputs = mech.CMI_single_step_forward(inputs)
        variable_loss_func(outputs, targets, output_variables)

        break


def test_forward():
    input_variables, output_variables, train_loader, valid_loader = prepare(freq_rate=1)

    mech = CMITestMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )

    mech.learn(train_loader, valid_loader, longest_epoch=10)
