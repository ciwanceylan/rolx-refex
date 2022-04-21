import numpy as np
import numba as nb
import scipy.sparse as sp


#
# @nb.jit(nb.float64[:, :](nb.types.ListType(nb.float64[:])), nopython=True, nogil=True)
# def mean_sum_agg(observations):
#     res = np.zeros((1, 2 * observations.shape[1]))
#
#     for i, val in enumerate(observations):
#         res[i, :] = np.asarray([np.mean(val, axis=0), np.sum(val, axis=0)])
#     return res
#
#
# @nb.jit(nb.float64[:, :](nb.types.ListType(nb.float64[:])), nopython=True, nogil=True)
# def basic_stats_agg(observations):
#     res = np.zeros((len(observations), 5))
#
#     for i, val in enumerate(observations):
#         res[i, :] = np.asarray([np.mean(val), np.sum(val),
#                                 np.min(val), np.max(val),
#                                 np.std(val)]
#                                )
#     return res
#
#
# @nb.jit(nb.float64[:, :](nb.types.ListType(nb.float64[:]), int), nopython=True, nogil=True)
# def percentile_agg_nb(observations, num_features: int):
#     q = np.linspace(0, 100, num_features)
#     res = np.zeros((len(observations), num_features))
#
#     for i, val in enumerate(observations):
#         res[i, :] = np.percentile(val, q=q, axis=0)
#     return res

def refex(adj: sp.spmatrix, use_weights=True, max_steps=5, prune_tol=0.0001):
    base_features = extract_base_features(adj, use_weights=use_weights)
    features = do_feature_recursion(base_features, adj, max_steps=max_steps, tol=prune_tol)
    return features


def adj2adj_dict(adj, return_weights=False, use_in_degrees=False, as_numba_dict=False):
    num_nodes = adj.shape[0]
    if use_in_degrees:
        adj = adj.tocsr()
    else:
        adj = adj.tocsc()

    adj.indices = adj.indices.astype(np.int64)
    adj.indptr = adj.indptr.astype(np.int64)

    if as_numba_dict:
        targets = nb.typed.Dict.empty(nb.int64, nb.int64[:])
        for i in range(num_nodes):
            targets[i] = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
    else:
        targets = {i: adj.indices[adj.indptr[i]:adj.indptr[i + 1]] for i in range(num_nodes)}

    if return_weights:
        adj.data = adj.data.astype(np.float64)
        if as_numba_dict:

            weights = nb.typed.Dict.empty(nb.int64, nb.float64[:])
            for i in range(num_nodes):
                weights[i] = adj.data[adj.indptr[i]:adj.indptr[i + 1]]
        else:
            weights = {i: adj.data[adj.indptr[i]:adj.indptr[i + 1]] for i in range(num_nodes)}
    else:
        weights = None
    return targets, weights


def extract_base_features(adj: sp.spmatrix, use_weights=True):
    node_feat = extract_node_features(adj, use_weights=use_weights)
    ego_feat = extract_egonet_features(adj, use_weights=use_weights)

    return np.concatenate((node_feat, ego_feat), axis=1)


def extract_node_features(adj: sp.spmatrix, use_weights=True):
    num_nodes = adj.shape[0]
    nz_row, nz_col = adj.nonzero()

    out_degrees_index, out_degrees_ = np.unique(nz_col, return_counts=True)
    out_degrees = np.zeros(num_nodes, dtype=np.float64)
    out_degrees[out_degrees_index] = out_degrees_

    in_degrees_index, in_degrees_ = np.unique(nz_row, return_counts=True)
    in_degrees = np.zeros(num_nodes, dtype=np.float64)
    in_degrees[in_degrees_index] = in_degrees_

    if use_weights:
        out_sum_weight = adj.sum(axis=0).A1
        in_sum_weight = adj.sum(axis=1).A1
        features = np.stack((out_degrees, in_degrees, out_sum_weight, in_sum_weight), axis=1)
    else:
        features = np.stack((out_degrees, in_degrees), axis=1)
    return features


def extract_egonet_features(adj, use_weights=True):
    out_adj_dict, out_weights = adj2adj_dict(adj, return_weights=use_weights, as_numba_dict=True)
    in_adj_dict, in_weights = adj2adj_dict(adj, use_in_degrees=use_weights, return_weights=True, as_numba_dict=True)

    if use_weights:
        features = _extract_egonet_features(out_adj_dict, in_adj_dict, out_weights, in_weights)
    else:
        features = _extract_egonet_features_no_weights(out_adj_dict, in_adj_dict)
    return features


@nb.jit(nb.float64[:, :](nb.types.DictType(nb.int64, nb.int64[:]), nb.types.DictType(nb.int64, nb.int64[:]),
                         nb.types.DictType(nb.int64, nb.float64[:]), nb.types.DictType(nb.int64, nb.float64[:])),
        nopython=True, nogil=True)
def _extract_egonet_features(out_adj_dict: nb.typed.Dict, in_adj_dict: nb.typed.Dict,
                             out_weights: nb.typed.Dict, in_weights: nb.typed.Dict):
    num_nodes = len(out_adj_dict)

    features = np.zeros((num_nodes, 6), dtype=np.float64)

    for v in range(num_nodes):
        egonet = set(np.unique(np.concatenate((out_adj_dict[v], in_adj_dict[v]))))
        egonet.add(v)
        num_internal_edges = 0
        num_out_edges = 0
        num_in_edges = 0

        internal_weight_sum = 0
        out_weight_sum = 0
        in_weight_sum = 0
        for neigh in egonet:
            for neigh_neigh, weight in zip(out_adj_dict[neigh], out_weights[neigh]):
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                    internal_weight_sum += weight
                else:
                    num_out_edges += 1
                    out_weight_sum += weight

            for neigh_neigh, weight in zip(in_adj_dict[neigh], in_weights[neigh]):
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                    internal_weight_sum += weight
                else:
                    num_in_edges += 1
                    in_weight_sum += weight

        features[v, 0] = num_internal_edges / 2
        features[v, 1] = num_out_edges
        features[v, 2] = num_in_edges

        features[v, 3] = internal_weight_sum / 2
        features[v, 4] = out_weight_sum
        features[v, 5] = in_weight_sum
    return features


@nb.jit(nb.float64[:, :](nb.types.DictType(nb.int64, nb.int64[:]), nb.types.DictType(nb.int64, nb.int64[:])),
        nopython=True, nogil=True)
def _extract_egonet_features_no_weights(out_adj_dict: nb.typed.Dict, in_adj_dict: nb.typed.Dict):
    num_nodes = len(out_adj_dict)

    features = np.zeros((num_nodes, 6), dtype=np.float64)

    for v in range(num_nodes):
        egonet = set(np.unique(np.concatenate((out_adj_dict[v], in_adj_dict[v]))))
        egonet.add(v)
        num_internal_edges = 0
        num_out_edges = 0
        num_in_edges = 0

        for neigh in egonet:
            for neigh_neigh in out_adj_dict[neigh]:
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                else:
                    num_out_edges += 1

            for neigh_neigh in in_adj_dict[neigh]:
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                else:
                    num_in_edges += 1

        features[v, 0] = num_internal_edges / 2
        features[v, 1] = num_out_edges
        features[v, 2] = num_in_edges
    return features


@nb.jit(nb.float64[:, :](nb.float64[:, :], nb.types.DictType(nb.int64, nb.int64[:]), nb.int64, nb.float64),
        nopython=True, nogil=True)
def _feature_recursion(features: np.ndarray, adj_dict: nb.typed.Dict, max_steps: int, tol: float):
    num_nodes = features.shape[0]
    for r in range(max_steps):
        num_new_features = 2 * features.shape[1]
        if features.shape[1] + num_new_features > num_nodes:
            break
        new_features = np.zeros((num_nodes, num_new_features))

        for v in range(num_nodes):
            num_neigh = len(adj_dict[v])
            if num_neigh == 0:
                continue
            new_features[v, :] = np.concatenate((np.sum(features[adj_dict[v], :], axis=0) / num_neigh,
                                                 np.sum(features[adj_dict[v], :], axis=0)))
        new_features = new_features / np.sqrt(np.sum(np.power(new_features, 2), axis=0))

        features = np.concatenate((features, new_features), axis=1)
        _, r = np.linalg.qr(features)
        dependence_coeff = np.abs(np.diag(r))
        keep_columns = dependence_coeff > tol * dependence_coeff.max()
        if np.all(keep_columns[-num_new_features:] == 0):
            break  # Break if no new features are added
        features = features[:, keep_columns]

    return features


def do_feature_recursion(features: np.ndarray, adj, max_steps: int, tol: float):
    adj_dict, _ = adj2adj_dict(adj.maximum(adj.T), as_numba_dict=True)
    return _feature_recursion(features, adj_dict, max_steps, tol)
