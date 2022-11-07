import torch
from torch.nn.functional import one_hot

from cmrl.models.networks.coder import VariableEncoder, VariableDecoder
from cmrl.utils.variables import ContinuousVariable, DiscreteVariable, BinaryVariable


def test_continuous_encoder():
    var_dim = 3
    output_dim = 5
    batch_size = 128

    var = ContinuousVariable(name="obs_0", dim=var_dim)

    encoder = VariableEncoder(var, output_dim, hidden_dims=[20])
    inputs = torch.rand(batch_size, var_dim)
    outputs = encoder(inputs)

    assert outputs.shape == (batch_size, output_dim)


def test_discrete_encoder():
    var_n = 3
    output_dim = 5
    batch_size = 128

    var = DiscreteVariable(name="obs_0", n=var_n)

    encoder = VariableEncoder(var, output_dim, hidden_dims=[20])
    inputs = one_hot(torch.randint(3, (batch_size,))).to(torch.float32)
    outputs = encoder(inputs)

    assert outputs.shape == (batch_size, output_dim)


def test_binary_encoder():
    output_dim = 5
    batch_size = 128

    var = BinaryVariable(name="obs_0")

    encoder = VariableEncoder(var, output_dim, hidden_dims=[20])
    inputs = one_hot(torch.randint(1, (batch_size,))).to(torch.float32)
    outputs = encoder(inputs)

    assert outputs.shape == (batch_size, output_dim)


def test_identity_decoder():
    var_dim = 3
    batch_size = 128

    var = ContinuousVariable(name="obs_0", dim=var_dim)

    decoder = VariableDecoder(var, identity=True)
    inputs = torch.rand(batch_size, var_dim * 2)
    outputs = decoder(inputs)

    assert outputs.shape == (batch_size, var_dim * 2)


def test_continuous_decoder():
    var_dim = 3
    input_dim = 5
    batch_size = 128

    var = ContinuousVariable(name="obs_0", dim=var_dim)

    decoder = VariableDecoder(var, input_dim, hidden_dims=[200])
    inputs = torch.rand(batch_size, input_dim)
    outputs = decoder(inputs)

    assert outputs.shape == (batch_size, var_dim * 2)


def test_discrete_decoder():
    var_n = 3
    input_dim = 5
    batch_size = 128

    var = DiscreteVariable(name="obs_0", n=var_n)

    decoder = VariableDecoder(var, input_dim, hidden_dims=[200])
    inputs = torch.rand(batch_size, input_dim)
    outputs = decoder(inputs)

    assert outputs.shape == (batch_size, var_n)
    assert torch.allclose(outputs.sum(dim=1), torch.tensor(1.0))


def test_binary_decoder():
    input_dim = 5
    batch_size = 128

    var = BinaryVariable(name="obs_0")

    decoder = VariableDecoder(var, input_dim, hidden_dims=[200])
    inputs = torch.rand(batch_size, input_dim)
    outputs = decoder(inputs)

    assert outputs.shape == (batch_size, 1)
    assert (outputs >= 0).all() and (outputs <= 1).all()
