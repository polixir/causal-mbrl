from omegaconf import DictConfig

from cmrl.models.networks.base_network import BaseNetwork


def test_base_network():
    network_cfg = DictConfig({"input_dim": 5, "output_dim": 10, "hidden_dims": [32, 32], "extra_dims": [7]})

    try:
        base_network = BaseNetwork(network_cfg, device="cpu")
        assert False
    except NotImplementedError:
        pass
