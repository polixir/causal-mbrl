import torch

from cmrl.models.graphs.prob_graph import BernoulliGraph


def test_init():
    g = BernoulliGraph(5, 5, include_input=True, init_param=0.1)
    expected = torch.ones(5, 5) * 0.1
    expected[torch.arange(5), torch.arange(5)] = BernoulliGraph._MASK_VALUE

    assert g.graph.size() == (5, 5)
    assert g.graph.equal(expected)


def test_adj_matrix():
    g = BernoulliGraph(5, 5, include_input=True, init_param=0.1)
    adj_mat = g.get_adj_matrix()
    expected = torch.sigmoid(torch.ones(5, 5) * 0.1)
    expected[torch.arange(5), torch.arange(5)] = 0

    assert adj_mat.equal(expected), "get_adj_matrix failed"

    binary_adj_matrix = g.get_binary_adj_matrix(0.5)
    expected = torch.ones(5, 5, dtype=torch.int)
    expected[torch.arange(5), torch.arange(5)] = 0

    assert binary_adj_matrix.equal(expected), "get_binary_adj_matrix failed"


def test_sample():
    g = BernoulliGraph(5, 5, include_input=True, init_param=0.1)
    samples = g.sample(None, 10, None)

    assert samples.size() == (10, 5, 5)
    assert samples[:, torch.arange(5), torch.arange(5)].any() == False
