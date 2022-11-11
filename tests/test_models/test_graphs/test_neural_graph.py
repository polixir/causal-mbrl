import torch
import torch.nn as nn

from cmrl.models.graphs.neural_graph import NeuralGraph


def test_init():
    g = NeuralGraph(5, 5, include_input=True)

    assert isinstance(g.graph, nn.Module)


def test_parameters():
    g = NeuralGraph(5, 5, include_input=True)
    p = g.parameters

    assert len(p) == len(list(g.graph.parameters()))


def test_adj_matrix():
    g = NeuralGraph(5, 5, include_input=True)
    assert next(g.graph.parameters()).grad is None

    inputs = torch.ones(2, 5)
    adj_mat = g.get_adj_matrix(inputs)

    assert adj_mat.size() == (2, 5, 5), "get_adj_matrix failed"
    b = adj_mat.sum()
    b.backward()
    assert next(g.graph.parameters()).grad is not None

    binary_adj_matrix = g.get_binary_adj_matrix(inputs, 0.5)

    assert binary_adj_matrix.size() == (2, 5, 5), "get_binary_adj_matrix failed"


def test_save_load():
    g = NeuralGraph(5, 5, include_input=True)
    g.save("./")
    old_graph = g.graph
    state_dict = old_graph.state_dict()

    g.load("./")
    assert g.graph is old_graph
    assert g.graph.state_dict() is not state_dict


if __name__ == "__main__":
    test_init()

    test_parameters()

    test_adj_matrix()

    test_save_load()
