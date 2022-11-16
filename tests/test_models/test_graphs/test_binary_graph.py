import os
import time
import shutil

import torch

from cmrl.models.graphs.binary_graph import BinaryGraph


def test_init():
    g = BinaryGraph(5, 5, include_input=True, init_param=1)

    assert g.graph.size() == (5, 5)
    assert g.graph.sum().item() == 20
    assert g.graph[torch.arange(5), torch.arange(5)].any().item() == False

    g = BinaryGraph(5, 5, extra_dim=3, init_param=torch.zeros(3, 5, 5))

    assert g.graph.size() == (3, 5, 5)
    assert g.graph.any().item() == False


def test_parameters():
    g = BinaryGraph(5, 5, include_input=True, init_param=1)
    params = g.parameters

    assert isinstance(params, tuple)
    assert len(params) == 1
    assert params[0].sum().item() == 20
    assert params[0][torch.arange(5), torch.arange(5)].any().item() == False

    g = BinaryGraph(5, 5, include_input=False, init_param=1)
    params = g.parameters

    assert params[0].all().item() == True


def test_adj_matrix():
    g = BinaryGraph(5, 5, include_input=True, init_param=1)
    adj_mat = g.get_adj_matrix()

    expected = torch.ones(5, 5, dtype=torch.int)
    expected[torch.arange(5), torch.arange(5)] = 0

    assert adj_mat.equal(expected), "get_adj_matrix failed"

    binary_adj_matrix = g.get_binary_adj_matrix()

    assert binary_adj_matrix.equal(expected), "get_binary_adj_matrix failed"


def test_set_data():
    g = BinaryGraph(5, 5, include_input=True, init_param=1)

    test_data = torch.ones(5, 5, dtype=torch.int)
    test_data[:2, :2] = 0

    expected = test_data.clone()
    expected[torch.arange(2, 5), torch.arange(2, 5)] = 0

    g.set_data(test_data)

    assert (g.graph == expected).all().item() == True


def test_save_load():
    # create a temp folder
    while True:
        save_dir = "./tmp" + str(time.time())
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            break

    g = BinaryGraph(5, 5, include_input=True, init_param=1)
    g.save(save_dir)
    old_graph = g.graph

    g.load(save_dir)
    assert g.graph is not old_graph
    assert g.graph.equal(old_graph)

    # clear the temp folder
    shutil.rmtree(save_dir)
