import gym
from stable_baselines3.common.buffers import ReplayBuffer
import torch
from torch.utils.data import DataLoader

from cmrl.models.causal_mech.reinforce import ReinforceCausalMech
from cmrl.models.data_loader import EnsembleBufferDataset, EnsembleBufferDataset, collate_fn
from cmrl.utils.creator import parse_space
from cmrl.utils.env import load_offline_data
from cmrl.models.causal_mech.util import variable_loss_func


def prepare(freq_rate):
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0", freq_rate=freq_rate, time_step=0.02)

    real_replay_buffer = ReplayBuffer(
        1000, env.observation_space, env.action_space, device="cpu", handle_timeout_termination=False
    )
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


def test_init():
    input_variables, output_variables, _, _ = prepare(freq_rate=1)

    mech = ReinforceCausalMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )

    assert mech.network, "network incorrectly initialized"
    assert mech.graph, "graph incorrectly initialized"
    assert mech.optimizer, "network optimizer incorrectly initialized"
    assert mech.graph_optimizer, "graph optimizer incorrectly initialized"


def test_causal_graph():
    input_variables, output_variables, _, _ = prepare(freq_rate=1)

    mech = ReinforceCausalMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )

    assert mech.causal_graph is not None, "cannot obtain causal graph correctly"
    assert mech.causal_graph.all(), "incorrect initial causal graph"


def test_single_step_forward():
    input_variables, output_variables, train_loader, _ = prepare(freq_rate=1)

    mech = ReinforceCausalMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )
    train_iter = iter(train_loader)
    inputs, targets = next(train_iter)

    # test training
    outputs = mech.single_step_forward(inputs, train=True)
    assert len(outputs.keys() ^ targets.keys()) == 0, "single forward output keys mismatch when training"
    loss = variable_loss_func(outputs, targets, mech.output_variables)
    assert loss is not None, "single forward output is incorrect when training"

    # test evaluating
    outputs = mech.single_step_forward(inputs, train=False)
    assert len(outputs.keys() ^ targets.keys()) == 0, "single forward output keys mismatch when evaluating"
    loss = variable_loss_func(outputs, targets, mech.output_variables)
    assert loss is not None, "single forward outupt is incorrect when evaluating"

    # test evaluating with fixed mask
    extra_dims = next(iter(inputs.values())).shape[:-1]
    mask = torch.randint(0, 2, size=(len(targets), *extra_dims, len(inputs)))
    outputs = mech.single_step_forward(inputs, train=False, mask=mask)
    assert len(outputs.keys() ^ targets.keys()) == 0, "single forward output keys mismatch when evaluating with given mask"
    loss = variable_loss_func(outputs, targets, mech.output_variables)
    assert loss is not None, "single forward output is incorrect when evaluating with given mask"


def test_forward():
    input_variables, output_variables, train_loader, _ = prepare(freq_rate=1)

    # test single step
    mech = ReinforceCausalMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )
    train_iter = iter(train_loader)
    inputs, targets = next(train_iter)

    outputs = mech.forward(inputs, train=True)
    assert len(outputs.keys() ^ targets.keys()) == 0, "forward output keys mismatch when evaluating (single step)"
    loss = variable_loss_func(outputs, targets, mech.output_variables)
    assert loss is not None, "forward outupt is incorrect when evaluating (single step)"

    # test multi-step
    mech = ReinforceCausalMech(
        name="test", input_variables=input_variables, output_variables=output_variables, multi_step="forward-euler 2"
    )
    train_iter = iter(train_loader)
    inputs, targets = next(train_iter)

    outputs = mech.forward(inputs, train=True)
    assert len(outputs.keys() ^ targets.keys()) == 0, "forward output keys mismatch when evaluating (2 steps)"
    loss = variable_loss_func(outputs, targets, mech.output_variables)
    assert loss is not None, "forward outupt is incorrect when evaluating (2 steps)"


def test_train_graph():
    input_variables, output_variables, train_loader, _ = prepare(freq_rate=1)

    # test single step
    mech = ReinforceCausalMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
    )

    grads = mech.train_graph(train_loader, data_ratio=1.0)
    assert grads is not None, "train graph single step failed"

    # test multi-step
    mech = ReinforceCausalMech(
        name="test",
        input_variables=input_variables,
        output_variables=output_variables,
        multi_step="forward-euler 2",
    )

    grads = mech.train_graph(train_loader, data_ratio=0.5)
    assert grads is not None, "train graph multi-step (2) failed"


def test_learn():
    input_variables, output_variables, train_loader, valid_loader = prepare(freq_rate=1)

    # test single step on cpu
    mech = ReinforceCausalMech(
        name="test single on cpu",
        input_variables=input_variables,
        output_variables=output_variables,
    )

    mech.learn(train_loader, valid_loader)

    # test multi-step on cpu
    mech = ReinforceCausalMech(
        name="test multi on cpu",
        input_variables=input_variables,
        output_variables=output_variables,
        multi_step="forward-euler 2",
    )

    mech.learn(train_loader, valid_loader)

    # test single step on cuda
    mech = ReinforceCausalMech(
        name="test single on cuda",
        input_variables=input_variables,
        output_variables=output_variables,
        device="cuda:0",
    )

    mech.learn(train_loader, valid_loader)

    # test multi-step on cuda
    mech = ReinforceCausalMech(
        name="test multi on cuda",
        input_variables=input_variables,
        output_variables=output_variables,
        multi_step="forward-euler 2",
        device="cuda:0",
    )

    mech.learn(train_loader, valid_loader)
