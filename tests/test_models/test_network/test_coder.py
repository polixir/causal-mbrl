import torch
import numpy as np
from torch.nn.functional import one_hot

from cmrl.models.networks.coder import VariableEncoder, VariableDecoder
from cmrl.types import Variable, ContinuousVariable, DiscreteVariable


def test_continuous_encoder():
    var_dim = 3
    node_dim = 5
    batch_size = 128

    var = ContinuousVariable(name="state0", dim=var_dim)

    encoder = VariableEncoder(var, node_dim)
    inputs = torch.rand(batch_size, var_dim)
    outputs = encoder(inputs)

    assert outputs.shape == (batch_size, node_dim)


def test_discrete_encoder():
    var_n = 3
    node_dim = 5
    batch_size = 128

    var = DiscreteVariable(name="state0", n=var_n)

    encoder = VariableEncoder(var, node_dim)
    inputs = one_hot(torch.randint(3, (batch_size,))).to(torch.float32)
    outputs = encoder(inputs)

    assert outputs.shape == (batch_size, node_dim)


def test_continuous_decoder():
    var_dim = 3
    node_dim = 5
    batch_size = 128

    var = ContinuousVariable(name="state0", dim=var_dim)

    decoder = VariableDecoder(var, node_dim)
    inputs = torch.rand(batch_size, node_dim)
    outputs = decoder(inputs)

    assert outputs.shape == (batch_size, var_dim)


def test_discrete_decoder():
    var_n = 3
    node_dim = 5
    batch_size = 128

    var = DiscreteVariable(name="state0", n=var_n)

    decoder = VariableDecoder(var, node_dim)
    inputs = torch.rand(batch_size, node_dim)
    outputs = decoder(inputs)

    assert outputs.shape == (batch_size, var_n)
    batch_sum = outputs.detach().numpy().sum(axis=1)
    assert np.allclose(batch_sum, 1)
