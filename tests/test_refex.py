import pytest
import networkx as nx
import numpy as np
import tempfile

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


@pytest.fixture()
def triplet_adj():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_node(3)
    graph.add_node(4)
    graph.add_node(5)
    return nx.adjacency_matrix(graph).T


@pytest.fixture()
def triplet_features():
    features = np.asarray([
        [1, 0],
        [-1, -1],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0]
    ])
    return features


@pytest.fixture()
def large_adj():
    graph = nx.fast_gnp_random_graph(1000, 0.01)
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


@pytest.mark.parametrize('tol', [-1, 0])
def test_recursion(triplet_adj, triplet_features, tol):
    features, _ = refex.do_feature_recursion(triplet_features, triplet_adj, max_steps=1, tol=tol)
    answer = np.asarray([
        [1, 0, -1, -1, -1, -1],
        [-1, -1, 1.5, 1, 3, 2],
        [2, 2, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    answer[:, 2:] = answer[:, 2:] / np.linalg.norm(answer[:, 2:], axis=0)
    if tol < 0:
        assert np.allclose(features, answer)
    else:
        assert np.allclose(features, answer[:, :3])


@pytest.mark.parametrize('tol', [-1, 1e-10, 1e-6, 0.001, 1.])
def test_refex(large_adj, tol):
    param = refex.RefexParam(max_emb_size=50, use_weights=False, prune_tol=tol)
    features, reprod_param = refex.refex(large_adj, param)

    if tol < 0:
        assert features.shape[0] == 1000
        assert features.shape[1] == 45
    else:
        assert features.shape[0] == 1000
        assert features.shape[1] <= 45
        assert features.shape[1] >= 5


@pytest.mark.parametrize('tol', [-1, 1e-10, 1e-6, 0.001, 1.])
def test_refex_reprod(large_adj, tol):
    param = refex.RefexParam(max_emb_size=128, use_weights=False, prune_tol=tol)
    features1, reprod_param = refex.refex(large_adj, param)
    with tempfile.NamedTemporaryFile() as fp:
        reprod_param.save(fp.name)
        fp.seek(0)
        param2 = refex.RefexParam.load(fp.name)

    features2, reprod_param = refex.refex(large_adj, param2)

    assert np.allclose(features1, features2)

    # if tol < 0:
    #     assert features.shape[0] == 1000
    #     assert features.shape[1] == 45
    # else:
    #     assert features.shape[0] == 1000
    #     assert features.shape[1] <= 45
    #     assert features.shape[1] >= 5
