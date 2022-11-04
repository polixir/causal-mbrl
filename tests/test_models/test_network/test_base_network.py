from omegaconf import DictConfig

from cmrl.models.networks.base_network import BaseNetwork


def test_base_network():
    try:
        base_network = BaseNetwork(device="cpu")
        assert False
    except NotImplementedError:
        pass
