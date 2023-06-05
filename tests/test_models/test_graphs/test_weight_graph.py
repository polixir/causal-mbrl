import os
import time
import shutil

import torch

from cmrl.models.graphs.weight_graph import WeightGraph


def test_init():
    g = WeightGraph(5, 5, include_input=True, init_param=0.5)
    expected = torch.ones(5, 5) * 0.5
    expected[torch.arange(5), torch.arange(5)] = 0

    assert g.graph.size() == (5, 5)
    assert g.graph.equal(expected)

    g = WeightGraph(5, 5, extra_dim=3, init_param=torch.ones(3, 5, 5) * 0.1)
    expected = torch.ones(3, 5, 5) * 0.1

    assert g.graph.size() == (3, 5, 5)
    assert g.graph.equal(expected)


def test_parameters():
    g = WeightGraph(5, 5, include_input=True, init_param=0.5)
    params = g.parameters
    expected = torch.ones(5, 5) * 0.5
    expected[torch.arange(5), torch.arange(5)] = 0

    assert isinstance(params, tuple)
    assert len(params) == 1
    assert params[0].equal(expected)

    g = WeightGraph(5, 5, include_input=False, init_param=0.1)
    params = g.parameters
    expected = torch.ones(5, 5) * 0.1

    assert params[0].equal(expected)


def test_adj_matrix():
    g = WeightGraph(5, 5, include_input=True, init_param=0.5)
    adj_mat = g.get_adj_matrix()
    expected = torch.ones(5, 5) * 0.5
    expected[torch.arange(5), torch.arange(5)] = 0

    assert adj_mat.equal(expected), "get_adj_matrix failed"

    binary_adj_matrix = g.get_binary_adj_matrix(0.4)
    expected = torch.ones(5, 5, dtype=torch.int)
    expected[torch.arange(5), torch.arange(5)] = 0

    assert binary_adj_matrix.equal(expected), "get_binary_adj_matrix failed"


def test_set_data():
    g = WeightGraph(5, 5, include_input=True, init_param=0.5)

    test_data = torch.ones(5, 5)
    test_data[:2, :2] = 0

    expected = test_data.clone()
    expected[torch.arange(2, 5), torch.arange(2, 5)] = 0

    g.set_data(test_data)

    assert (g.graph == expected).all().item() == True


def test_grad():
    g = WeightGraph(5, 5, init_param=0.5, requires_grad=False)
    assert g.requries_grad == False, "no grad test failed"

    g = WeightGraph(5, 5, init_param=0.5, requires_grad=True)
    assert g.requries_grad, "grad test, requires_grad failed"
    assert g.graph.grad is None, "grad test, grad is None"

    c = g.parameters[0].abs().sum()
    c.backward()
    expected = torch.ones(5, 5)
    assert g.graph.grad is not None and g.graph.grad.equal(expected), "grad test, incorrect grad"

    g = WeightGraph(5, 5, init_param=0.5, include_input=True, requires_grad=True)
    assert g.graph.grad is None, "grad test, include input, requires_grad failed"

    c = g.parameters[0].abs().sum()
    c.backward()
    expected = torch.ones(5, 5)
    expected[torch.arange(5), torch.arange(5)] = 0
    assert g.graph.grad is not None and g.graph.grad.equal(expected), "grad test, include input, incorrect grad"

    g = WeightGraph(5, 5, init_param=0.5, include_input=True, requires_grad=True)
    p = g.parameters[0]
    test_data = torch.ones(5, 5)
    g.set_data(test_data)

    c = p.abs().sum()
    c.backward()
    expected = torch.ones(5, 5)
    expected[torch.arange(5), torch.arange(5)] = 0
    assert g.graph.grad is not None and g.graph.grad.equal(expected), "grad test, include input, set_data, incorrect grad"


def test_save_load():
    # create a temp folder
    while True:
        save_dir = "./tmp" + str(time.time())
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            break

    g = WeightGraph(5, 5, include_input=True, init_param=0.5)
    g.save(save_dir)
    old_graph = g.graph

    g.load(save_dir)
    assert g.graph is not old_graph
    assert g.graph.equal(old_graph)

    # clear the temp folder
    shutil.rmtree(save_dir)
