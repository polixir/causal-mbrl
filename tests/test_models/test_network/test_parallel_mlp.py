from omegaconf import DictConfig
import torch

from cmrl.models.networks.parallel_mlp import ParallelMLP


def test_parallel_mlp():
    input_dim = 5
    output_dim = 6
    use_bias = True
    extra_dims = [7]
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    network_cfg = dict(
        {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": [32, 32],
            "bias": use_bias,
            "extra_dims": extra_dims,
            "activation_fn_cfg": DictConfig({"_target_": "torch.nn.SiLU"}),
        }
    )

    mlp = ParallelMLP(**network_cfg).to(device)

    model_in = torch.rand((batch_size, input_dim)).to(device)
    model_out = mlp(model_in)
    assert model_out.shape == (
        *extra_dims,
        batch_size,
        output_dim,
    )

    assert str(mlp.device).startswith(device)
