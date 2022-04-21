import numpy as np
import numba as nb
import scipy.sparse as sp


def refex(adj: sp.spmatrix, use_weights=True, max_steps=5, prune_tol=0.0001):
    """ Computes ReFeX features.
    Feature pruning uses QR factorisation and removes feature columns for which the diagonal element of R is smaller
    than `prune_tol * diag(R).max()`.

    Args:
        adj: Weighted adjacency matrix with A[i,j] indicating an edge j -> i. Assumes non-negative weights.
        use_weights: Append features using weights
        max_steps: Maximum number of recursion steps.
        prune_tol: The feature pruning tolerance.
    Returns:
        features: Matrix of features. [n x 10] if use_weights, else  [n x 5]
    """
    base_features = extract_base_features(adj, use_weights=use_weights)
    features = do_feature_recursion(base_features, adj, max_steps=max_steps, tol=prune_tol)
    return features


def extract_base_features(adj: sp.spmatrix, use_weights=True):
    """ Extract node and egonet features for each node in the graph.

    Args:
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        use_weights: Append features using weights
    Returns:
        features: Matrix of features. [n x 10] if use_weights, else  [n x 5]
    """
    node_feat = extract_node_features(adj, use_weights=use_weights)
    ego_feat = extract_egonet_features(adj, use_weights=use_weights)

    return np.concatenate((node_feat, ego_feat), axis=1)


def extract_node_features(adj: sp.spmatrix, use_weights=True):
    """ Extract node features for each node in the graph. Node features are in and out degrees, and their weighted
    counterparts is `use_weights`.

    Args:
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        use_weights: Append weighted node degrees
    Returns:
        features: Matrix of node features. [n x 4] if use_weights, else  [n x 2]
    """
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
    """ Extract egonet features for each node in the network.
    The egonet features are the number of edges in the egonet and the number of incoming and outgoing edges of the
    egonet. Sum of weights are used if `use_weights` is `True`.


    Args:
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        use_weights: Append weighted egonet features.

    Returns:
        features: Egonet features. [n x 6] if use_weights else [n x 3]
    """
    out_adj_dict, out_weights = adj2adj_dict(adj, return_weights=use_weights, as_numba_dict=True)
    in_adj_dict, in_weights = adj2adj_dict(adj, use_in_degrees=True, return_weights=use_weights, as_numba_dict=True)

    if use_weights:
        features = _extract_egonet_features(out_adj_dict, in_adj_dict, out_weights, in_weights)
    else:
        features = _extract_egonet_features_no_weights(out_adj_dict, in_adj_dict)
    return features


def do_feature_recursion(features: np.ndarray, adj: sp.spmatrix, max_steps: int, tol: float):
    """ Enhance features through recursive aggregation using mean and sum of neighbour features.
    Features are only added if they are linearly independent of existing features.
    Args:
        features: Base features to enhance
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        max_steps: The maximum number of recursion steps
        tol: Tolerance for adding new features

    Returns:
        features: Matrix of enhanced features, [n x *]
    """
    adj_dict, _ = adj2adj_dict(adj.maximum(adj.T), as_numba_dict=True)
    return _feature_recursion(features.astype(np.float64), adj_dict, max_steps, tol)


def adj2adj_dict(adj: sp.spmatrix, return_weights=False, use_in_degrees=False, as_numba_dict=False):
    """ Get a adjacency dictionary representation of the graph defined by adj.

    Args:
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        return_weights: Return a dictionary of the weights associated with each edge.
        use_in_degrees: Create the adjacency dict using incoming edges instead of outgoing.
        as_numba_dict: Create a numba Dict which can be used with nopython numba functions.

    Returns:
        targets: Dict of adjacent nodes for each node in the graph
        weights: Dict of edge weights for adjacent node. None if return_weights=False.
    """
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

    features = np.zeros((num_nodes, 3), dtype=np.float64)

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
        nopython=True, nogil=True, parallel=True)
def _feature_recursion(features: np.ndarray, adj_dict: nb.typed.Dict, max_steps: int, tol: float):
    num_nodes = features.shape[0]
    for r in range(max_steps):
        num_new_features = 2 * features.shape[1]
        if features.shape[1] + num_new_features > num_nodes:
            break
        new_features = np.zeros((num_nodes, num_new_features))

        for v in nb.prange(num_nodes):
            num_neigh = len(adj_dict[v])
            if num_neigh == 0:
                continue
            new_features[v] = np.concatenate((np.sum(features[adj_dict[v], :], axis=0) / num_neigh,
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
