from unittest import TestCase

import torch
from torch.nn import Linear

from cmrl.models.layers import ParallelLinear


def test_origin_layer():
    input_dim = 5
    output_dim = 6
    bias = True
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = ParallelLinear(
        input_dim=input_dim,
        output_dim=output_dim,
        bias=bias,
    ).to(device)

    model_in = torch.rand((batch_size, input_dim)).to(device)
    model_out = layer(model_in)
    assert model_out.shape == (
        batch_size,
        output_dim,
    )


def test_one_extra_dims_linear():
    input_dim = 5
    output_dim = 6
    bias = True
    extra_dims = [7]
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = ParallelLinear(
        input_dim=input_dim,
        output_dim=output_dim,
        bias=bias,
        extra_dims=extra_dims,
    ).to(device)

    model_in = torch.rand((*extra_dims, batch_size, input_dim)).to(device)
    model_out = layer(model_in)
    assert model_out.shape == (
        extra_dims[0],
        batch_size,
        output_dim,
    )


def test_two_extra_dims_linear():
    input_dim = 5
    output_dim = 1
    bias = True
    extra_dims = [6, 7]
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = ParallelLinear(
        input_dim=input_dim,
        output_dim=output_dim,
        bias=bias,
        extra_dims=extra_dims,
    ).to(device)

    model_in = torch.rand((*extra_dims, batch_size, input_dim)).to(device)
    model_out = layer(model_in)
    assert model_out.shape == (
        extra_dims[0],
        extra_dims[1],
        batch_size,
        output_dim,
    )


def test_broadcast_two_extra_dims_linear():
    input_dim = 5
    output_dim = 1
    bias = True
    extra_dims = [6, 7]
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = ParallelLinear(
        input_dim=input_dim,
        output_dim=output_dim,
        bias=bias,
        extra_dims=extra_dims,
    ).to(device)

    broadcast_dim = 10

    model_in = torch.rand((broadcast_dim, *extra_dims, batch_size, input_dim)).to(device)
    model_out = layer(model_in)
    assert model_out.shape == (
        broadcast_dim,
        extra_dims[0],
        extra_dims[1],
        batch_size,
        output_dim,
    )


def test_broadcast_linear():
    input_dim = 5
    output_dim = 1
    bias = True
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = Linear(input_dim, output_dim, bias=bias).to(device)

    broadcast_dim = 10

    model_in = torch.rand((broadcast_dim, batch_size, input_dim)).to(device)
    model_out = layer(model_in)
    assert model_out.shape == (
        broadcast_dim,
        batch_size,
        output_dim,
    )


def test_repr():
    layer = ParallelLinear(3, 5)
    print(repr(layer))
    assert True


def test_device():
    layer = ParallelLinear(3, 5).to("cpu")
    assert str(layer.device) == "cpu"
