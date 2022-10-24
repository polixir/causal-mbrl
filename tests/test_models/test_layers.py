from unittest import TestCase

import torch

from cmrl.models.layers import ParallelLinear


def test_origin_layer():
    input_dim = 5
    output_dim = 6
    use_bias = True
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = ParallelLinear(
        input_dim=input_dim,
        output_dim=output_dim,
        use_bias=use_bias,
    ).to(device)

    model_in = torch.rand((batch_size, input_dim)).to(device)
    model_out = layer(model_in)
    assert model_out.shape == (
        batch_size,
        output_dim,
    )


def test_two_extra_dims_layer():
    input_dim = 5
    output_dim = 6
    use_bias = True
    extra_dims = [3, 4]
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = ParallelLinear(
        input_dim=input_dim,
        output_dim=output_dim,
        use_bias=use_bias,
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


def test_repr():
    layer = ParallelLinear(3, 5)
    print(repr(layer))
    assert True


def test_device():
    layer = ParallelLinear(3, 5).to("cpu")
    assert str(layer.device) == "cpu"
