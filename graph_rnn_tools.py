import numpy as np
import networkx as nx
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_sequence, pad_packed_sequence
from collections import namedtuple
from torch.utils.data.dataset import Dataset
from itertools import product
from collections import deque
import random
from random import shuffle
SCALE_SPATIAL = 1e4


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output


def bfs_handmade(G, start):
    visited, queue = set(), deque([start])
    bfs = []
    while queue:
        vertex = queue.popleft()
        bfs.append(vertex)
        edges = [x[1] for x in G.edges(vertex, data=True)];
        shuffle(edges)
        # edges = sorted(G.edges(vertex, data=True), key=lambda x: x[2]['weight']); edges = [x[1] for x in edges]
        for neighbour in edges:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return bfs


def encode_adj(adj, max_prev_node=10, is_full=False):
    '''
    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0] - 1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n - 1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1) * (n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order

    return adj_output


def encode_adj(adj, max_prev_node=10, is_full=False):
    '''
    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0] - 1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n - 1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1) * (n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order

    return adj_output


def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output


def decode_adj_flexible(adj_output):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i+1-len(adj_output[i])
        output_end = i+1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output)+1, len(adj_output)+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


"""
def drop_nodes(G, probability=0.1):
    nodes = list(G.nodes())[1:]
    droped_nodes = np.random.choice(nodes, size=int(probability * len(nodes)), replace=False)
    for node in droped_nodes:
        G.add_edges_from(list(product(list(G.predecessors(node)), list(G.successors(node)))))
        G.remove_node(node)
    return G, droped_nodes
"""


def drop_nodes(G, droped_nodes):
    for node in droped_nodes:
        G.add_edges_from(list(product(list(G.predecessors(node)), list(G.successors(node)))))
        G.remove_node(node)
    return G


def shower_rotation(X):
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    X[:, [1, 2]] = X[:, [1, 2]].dot(R)
    X[:, [4, 5]] = X[:, [4, 5]].dot(R)
    return X


shower_adj = namedtuple('shower_adj', field_names=['x', 'adj', 'ele_p'])
graphrnn_shower = namedtuple('graphrnn_shower', field_names=['x',
                                                             'diff_x',
                                                             'adj',
                                                             'adj_out',
                                                             'node_weights',
                                                             'node_in_degree',
                                                             'node_out_degree',
                                                             'node_order',
                                                             'ele_p',
                                                             'edge_weights'])


def preprocess_shower_for_graphrnn(shower,
                                   device,
                                   max_prev_node=8,
                                   q=0.1,
                                   debug=False):
    X = np.vstack([
        np.arange(len(shower.x.SX)),
        shower.x.SX,
        shower.x.SY,
        shower.x.SZ,
        shower.x.TX,
        shower.x.TY]
    ).T.copy()

    x_shower = np.vstack([
        shower.x.SX,
        shower.x.SY,
        shower.x.SZ,
        shower.x.TX,
        shower.x.TY]
    ).T.copy()

    low_tx, high_tx = np.percentile(x_shower[:, 3], [q, 100 - q])
    low_ty, high_ty = np.percentile(x_shower[:, 4], [q, 100 - q])
    low_sx, high_sx = np.percentile(x_shower[:, 0], [q, 100 - q])
    low_sy, high_sy = np.percentile(x_shower[:, 1], [q, 100 - q])

    mask = (
            (low_tx < x_shower[:, 3]) & (x_shower[:, 3] < high_tx) &
            (low_ty < x_shower[:, 4]) & (x_shower[:, 4] < high_ty) &
            (low_sx < x_shower[:, 0]) & (x_shower[:, 0] < high_sx) &
            (low_sy < x_shower[:, 1]) & (x_shower[:, 1] < high_sy)
    )

    X = shower_rotation(X)

    # G = nx.DiGraph(shower.adj)
    # not symmetric by construction(i.e.) tree
    # need symmetrization
    G = nx.bfs_tree(nx.Graph(shower.adj), 0)
    G = drop_nodes(G, np.where(~mask)[0])
    X = X[mask]
    # print(list(nx.bfs_edges(G, 0)))
    # G, droped_nodes = drop_nodes(G, probability=probability)
    # mask = np.ones(len(X), bool)
    # mask[droped_nodes] = False
    # X = X[mask]
    # G = nx.Graph(G)  # track dropout
    adj = np.asarray(nx.to_numpy_matrix(G))
    # here is ok to be DiGraph, becuase in bfs_handmade used edges
    G = nx.DiGraph(adj)
    # add randomness in bfs?
    start_idx = 0

    x_idx = np.array(bfs_handmade(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]

    G = nx.bfs_tree(nx.Graph(adj), 0)
    # print(list(nx.bfs_edges(nx.Graph(adj), 0)))
    node_degrees = dict(G.degree())
    node_in_degree = list(G.in_degree())
    node_in_degree = sorted(node_in_degree, key=lambda x: x[0])
    node_in_degree = torch.tensor([nid[1] for nid in node_in_degree], dtype=torch.float32)  # .to(device)

    node_out_degree = list(G.out_degree())
    node_out_degree = sorted(node_out_degree, key=lambda x: x[0])
    node_out_degree = torch.tensor([nod[1] for nod in node_out_degree], dtype=torch.float32)  # .to(device)

    node_order = torch.tensor(list(G.nodes())).float() / 100. # .to(device)

    edge_weights = []
    for i, j in G.edges():
        edge_weights.append(np.exp(-2 * np.abs((node_degrees[i] + node_degrees[j]) / 4 - 1)))

    node_weights = [d for n, d in G.degree()]
    node_weights = torch.tensor(node_weights, dtype=torch.float32).view(-1)  # .to(device)

    # actual data
    adj_output = encode_adj(adj.T, max_prev_node=max_prev_node)

    if debug:
        adj_recover = decode_adj(adj_output)
        return adj_recover, adj, adj_output

        return np.abs((adj_recover - adj)).sum()

    X = X[x_idx, 1:]
    X = X / np.array([SCALE_SPATIAL, SCALE_SPATIAL, SCALE_SPATIAL, 1, 1])

    # for now forget about distances
    # TODO: what to do with distances?
    adj_output[adj_output != 0] = 1.

    adj_output_t = torch.tensor(
        np.append(np.append(np.ones((1, max_prev_node)), adj_output, axis=0), np.zeros((1, max_prev_node)), axis=0),
                                dtype=torch.float32).view(1, -1, max_prev_node)  # .to(device)

    X_t = torch.tensor(X, dtype=torch.float32).view(1, -1, 5)  # .to(device)
    
    adj_decoded = decode_adj(adj_output)

    # adj in/out
    adj_decoded[np.tril_indices_from(adj_decoded)] = 0
    G = nx.Graph(adj_decoded)
    edges = list(nx.bfs_edges(G, 0))

    adj_out_t = torch.LongTensor(np.array(edges).T) # .to(device)
    diff_x = X_t[0][adj_out_t[1]] - X_t[0][adj_out_t[0]]
    return graphrnn_shower(adj=adj_output_t,
                           x=X_t,
                           diff_x=diff_x,
                           node_weights=node_weights,
                           adj_out=adj_out_t,
                           node_in_degree=node_in_degree.view(-1, 1),
                           node_out_degree=node_out_degree.view(-1, 1),
                           node_order=node_order.view(-1, 1),
                           edge_weights=torch.tensor(edge_weights,
                                                     dtype=torch.float32).view(-1, 1),  #.to(device),
                           ele_p=torch.tensor(shower.ele_p, dtype=torch.float32)  # .to(device)
    )


class TorchShowers(Dataset):
    def __init__(self, showers, max_prev_node, device, q):
        self.showers = showers
        self.max_prev_node = max_prev_node
        self.q = q
        self.device = device

    def __getitem__(self, index):
        return preprocess_shower_for_graphrnn(
            self.showers[index],
            max_prev_node=self.max_prev_node,
            device=self.device,
            q=self.q
        )

    def __len__(self):
        return len(self.showers)

