import pytest
import networkx as nx
import numpy as np

import rolxrefex.refex as refex


@pytest.fixture()
def five_node_graph_adj():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(2, 1)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(2, 4)
    graph.add_edge(1, 4)
    graph.add_edge(4, 3)
    graph.add_edge(4, 2)
    graph.add_edge(4, 1)
    return nx.adjacency_matrix(graph).T


@pytest.mark.parametrize('use_weights', [True, False])
def test_egonet_features(five_node_graph_adj, use_weights):
    ego_features = refex.extract_egonet_features(five_node_graph_adj, use_weights=use_weights)
    answer = np.asarray([[6., 3., 3., 6., 3., 3.],
                         [8., 3., 1., 8., 3., 1.],
                         [12., 0., 0., 12., 0., 0.],
                         [7., 3., 2., 7., 3., 2.],
                         [9., 0., 3., 9., 0., 3.]])
    if use_weights:
        assert np.allclose(ego_features, answer)
    else:
        assert np.allclose(ego_features, answer[:, :3])


@pytest.mark.parametrize('use_weights', [True, False])
def test_node_features(five_node_graph_adj, use_weights):
    node_features = refex.extract_node_features(five_node_graph_adj, use_weights=use_weights)

    answer = np.asarray([[3., 0., 3., 0.],
                         [2., 3., 2., 3.],
                         [3., 3., 3., 3.],
                         [1., 3., 1., 3.],
                         [3., 3., 3., 3.]])
    if use_weights:
        assert np.allclose(node_features, answer)
    else:
        assert np.allclose(node_features, answer[:, :2])
