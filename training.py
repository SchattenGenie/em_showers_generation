import comet_ml
from comet_ml import Experiment
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from graph_rnn_tools import TorchShowers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential
from torch.distributions import Bernoulli
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import PackedSequence
import pyro
import pyro.distributions as dist
from itertools import product
import pickle
from collections import deque
from random import shuffle
from graph_rnn import GraphRNN
from features_nn import FeaturesGCN
from tqdm import tqdm
from collections import namedtuple
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_sequence, pad_packed_sequence
import torch_geometric.transforms as T
from graph_rnn_tools import decode_adj
import networkx as nx
import torch_cluster
import torch_geometric
import os
import click


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
SCALE_SPATIAL = 1e4


def collate_fn(x):
    return x


def concat_padded_tensors(tensor, lengths):
    return torch.cat([tensor[i, :length, :][1:, :] for i, length in enumerate(lengths)], dim=0)


def process_train_graphrnn(showers_batch, model, edge_nn, device):
    batch_size = len(showers_batch)

    model.hidden = model.init_hidden(input=torch.stack([x.ele_p for x in showers_batch]).view(-1, 1),
                                     batch_size=batch_size)

    # x.adj = [N, max_nodes]
    # packed_adj_batch = [batch_size, N_max, max_nodes]
    packed_adj_batch = pack_sequence([x.adj[0] for x in showers_batch])

    # embedding_batch -- padded [batch_size, N_max, hidden_size]
    embedding_batch_raw, embedding_batch, output_len = model(packed_adj_batch, pack=True)

    # shower embedding_batch[i, :output_len[i], :]
    packed_embedding_batch = concat_padded_tensors(embedding_batch, output_len)
    packed_adj_batch = torch.cat([x.adj[0][1:] for x in showers_batch], dim=0)

    hidden_null = torch.zeros(edge_nn.num_layers - 1,
                              packed_embedding_batch.shape[0],
                              packed_embedding_batch.shape[1]).to(device)

    edge_nn.hidden = torch.cat((packed_embedding_batch.view(
        1,
        packed_embedding_batch.size(0),
        packed_embedding_batch.size(1)
    ), hidden_null), dim=0)

    # packed_adj_batch_data = [total_edges, max_num_nodes, 1] // [batch_size, seq_length, input_size]
    packed_adj_batch = packed_adj_batch.view(packed_adj_batch.shape[0],
                                             packed_adj_batch.shape[1], 1)

    #
    packed_adj_batch = torch.cat((torch.ones(packed_adj_batch.shape[0], 1, 1).to(device),
                                  packed_adj_batch), dim=1)
    edges_emb, edges, _ = edge_nn(packed_adj_batch)

    loss_bce = nn.BCELoss(reduction='none').to(device)
    loss_bc = loss_bce(torch.sigmoid(edges[:, :-1, :]), packed_adj_batch[:, 1:, :])
    weights = len(packed_adj_batch[:, 1:, :]) / (1 + packed_adj_batch[:, 1:, :].sum(dim=0))
    weights = (len(weights) * weights / weights.sum())[:, 0]
    loss_bc = (loss_bc.mean(dim=0).mean(dim=1) * weights).sum()

    return embedding_batch_raw, \
           output_len, \
           loss_bc


DISTANCE = 1293. / SCALE_SPATIAL
EPS = 1e-5


def opera_distance_metric(basetrack_left, basetrack_right):
    # x, y, z, tx, ty
    # 0, 1, 2, 3, 4
    mask_swap = (basetrack_right[:, 2] < basetrack_left[:, 2])
    basetrack_right[mask_swap], basetrack_left[mask_swap] = basetrack_left[mask_swap], basetrack_right[mask_swap]
    dz = basetrack_right[:, 2] - basetrack_left[:, 2]

    dx = basetrack_left[:, 0] - (basetrack_right[:, 0] - basetrack_right[:, 3] * dz)
    dy = basetrack_left[:, 1] - (basetrack_right[:, 1] - basetrack_right[:, 4] * dz)

    dtx = basetrack_left[:, 3] - basetrack_right[:, 3]
    dty = basetrack_left[:, 4] - basetrack_right[:, 4]

    # dz = DISTANCE
    a = (dtx).pow(2) + (dty).pow(2)
    b = 2 * (dtx * dx + dty * dy)
    c = dx.pow(2) + dy.pow(2)

    mask = (a == torch.tensor(0., dtype=torch.float32).to(basetrack_left.device))
    result = torch.zeros_like(a)
    result[mask] = (torch.abs(torch.sqrt(c)) * dz / DISTANCE)[mask]

    a = a[~mask]
    b = b[~mask]
    c = c[~mask]
    discriminant = (b ** 2. - 4. * a * c)
    log_denominator = 2. * torch.sqrt(a) * torch.sqrt(torch.abs((a * dz + b) * dz + c)) + 2 * a * dz + b + EPS
    log_numerator = 2. * torch.sqrt(a) * torch.sqrt(c) + b + EPS
    first_part = ((2. * a * dz + b) * torch.sqrt(torch.abs(dz * (a * dz + b) + c)) - b * torch.sqrt(c)) / (4. * a)
    result[~mask] = torch.abs((discriminant * torch.log(torch.abs(log_numerator / log_denominator)) / (
                8. * torch.sqrt(a * a * a)) + first_part)) / DISTANCE

    return result


def loss_mse_edges(showers_x, edge_index, features):
    loss_mse = torch.nn.MSELoss()
    return loss_mse((showers_x[edge_index[0]] - showers_x[edge_index[1]]),
                    features)


def loss_edges(edge_index, features, add_dim):
    x = features
    predictions = torch.zeros(len(x) + add_dim, 5).to(x.device)
    for i, (j, k) in enumerate(edge_index.t()):
        predictions[k] = predictions[j] + x[i]
    node_from = torch.index_select(predictions, dim=0, index=edge_index[0])
    node_to = torch.index_select(predictions, dim=0, index=edge_index[1])
    smoothness = torch.mean(opera_distance_metric(node_from, node_to))
    loss_mse = torch.nn.MSELoss()
    IP_left = loss_mse(node_from[:, 0:2] * node_from[:, 3:], node_to[:, :2])
    IP_right = loss_mse(node_to[:, 0:2] * node_to[:, 3:], node_from[:, :2])
    # TODO: special loss!
    return smoothness, IP_left, IP_right


def loss_edges_over_predictions(edge_index, predictions):
    node_from = torch.index_select(predictions, dim=0, index=edge_index[0])
    node_to = torch.index_select(predictions, dim=0, index=edge_index[1])
    smoothness = torch.mean(opera_distance_metric(node_from, node_to))
    loss_mse = torch.nn.MSELoss()
    IP_left = loss_mse(node_from[:, 0:2] * node_from[:, 3:], node_to[:, :2])
    IP_right = loss_mse(node_to[:, 0:2] * node_to[:, 3:], node_from[:, :2])
    # TODO: special loss!
    return smoothness, IP_left, IP_right


def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


def generate_graph(model, edge_nn, features_nn, max_prev_node, test_batch_energies, device):
    test_batch_size = test_batch_energies.shape[0]

    # hidden_state = [num_layers, batch_size, hidden_size]
    model.train()
    edge_nn.train()
    features_nn.train()

    # generate graphs
    max_num_node = 200

    # предсказания нейронки сэмплированные с помощью бернулли
    # y_pred = [batch_size, seq_length, max_nodes]
    y_pred_long = torch.ones(test_batch_size,
                             max_num_node,
                             max_prev_node).to(device)  # discrete prediction

    # x_step = [batch_size, 1, max_nodes]
    # x_step = [batch_size, seq_length, max_nodes]
    # seq_length == 1
    model.hidden = model.init_hidden(test_batch_energies, test_batch_size)
    x_step = torch.ones(test_batch_size, 1, max_prev_node).to(device)
    embs = []
    for i in range(max_num_node):
        # output_raw_emb, output_raw, output_len
        # h = [test_batch, seq_length, output_size] # seq_length = 1
        emb, h, _ = model(x_step)
        embs.append(emb)
        # hidden_state = [num_layers, batch_size, hidden_size]
        hidden_null = torch.zeros(edge_nn.num_layers - 1, h.size(0), h.size(2)).to(device)
        edge_nn.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)  # num_layers, batch_size, hidden_size
        # sampling x_step from output h
        x_step = torch.zeros(test_batch_size, 1, max_prev_node).to(device)
        output_x_step = torch.ones(test_batch_size, 1, 1).to(device)
        for j in range(max_prev_node):
            # edge_emb = [batch_size, seq_length, hidden_size]
            # output_y_pred_step = [batch_size, seq_length, 1]
            edge_emb, output_y_pred_step, _ = edge_nn(output_x_step)
            if j < i + 1:
                output_x_step = Bernoulli(probs=torch.sigmoid(output_y_pred_step)).sample()
                x_step[:, :, j:j + 1] = output_x_step
        y_pred_long[:, i:i + 1, :] = x_step
    # TODO: need?
    emb, h, _ = model(x_step)
    embs.append(emb)

    embs = torch.cat(embs, dim=1)

    y_pred_long_data = y_pred_long.data.long()
    showers = []

    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].detach().cpu().numpy())
        emb = embs[i]
        # G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred = nx.from_numpy_matrix(adj_pred)  # get a graph from zero-padded adj
        G_pred = max(nx.connected_component_subgraphs(G_pred), key=len)
        nodes = list(G_pred.nodes())
        G_pred = nx.bfs_tree(G_pred, nodes[0])
        adj_pred = nx.adjacency_matrix(G_pred).toarray()
        nodes = list(G_pred.nodes())
        emb = emb[nodes]
        G_pred = nx.DiGraph(adj_pred)

        # for nodes with
        node_in_degree = list(G_pred.in_degree())
        node_in_degree = sorted(node_in_degree, key=lambda x: x[0])
        node_in_degree = torch.tensor([nid[1] for nid in node_in_degree], dtype=torch.float32).to(device).view(-1, 1)

        node_out_degree = list(G_pred.out_degree())
        node_out_degree = sorted(node_out_degree, key=lambda x: x[0])
        node_out_degree = torch.tensor([nod[1] for nod in node_out_degree], dtype=torch.float32).to(device).view(-1, 1)

        node_order = torch.tensor(list(G_pred.nodes())).float().to(device).view(-1, 1) / 100.
        if len(G_pred) <= 2:
            continue

        edges = list(nx.bfs_edges(G_pred, 0))
        adj_out = torch.LongTensor(np.array(edges).T).to(device)
        shower_t = torch_geometric.data.Data(x=torch.cat([emb, node_in_degree, node_out_degree, node_order], dim=1),
                                             edge_index=adj_out).to(device)

        # GCN to recover shower features
        x = features_nn.generate(shower_t)
        predictions = torch.zeros(len(shower_t.x), 5).to(device)
        for i, (j, k) in enumerate(shower_t.edge_index.t()):
            predictions[k] = predictions[j] + x[i]

        shower_t = torch_geometric.data.Data(x=predictions,
                                             edge_index=shower_t.edge_index).to(device)
        showers.append(shower_t)

    return showers


def train_epoch(model, edge_nn, features_nn, dataset_loader, optimizer, device, max_prev_node, loss_weights=None, signal_gen=True):
    if loss_weights is None:
        loss_weights = defaultdict(lambda: 1.)
    model.train()
    edge_nn.train()
    features_nn.train()
    metrics = defaultdict(list)

    ll_logits = torch.tensor(-1)
    ll_IP_right = torch.tensor(-1)
    ll_IP_left = torch.tensor(-1)
    ll_smoothness = torch.tensor(-1)
    ll_mse = torch.tensor(-1)
    ll_smoothness_gen = torch.tensor(-1)
    ll_IP_left_gen = torch.tensor(-1)
    ll_IP_right_gen = torch.tensor(-1)

    for showers_batch in tqdm(dataset_loader):
        showers_batch = [shower_to_device(shower, device=device) for shower in showers_batch]
        optimizer.zero_grad()
        showers_batch = sorted(showers_batch, key=lambda x: -x.adj.shape[1])

        embedding_batch, output_len, ll_bce = process_train_graphrnn(showers_batch,
                                                                     model=model,
                                                                     edge_nn=edge_nn,
                                                                     device=device)
        if signal_gen:
            # iterate over showers in batch and calc losses
            showers_t = []
            showers_x = []
            for k, l in enumerate(output_len):
                shower = showers_batch[k]
                embedding = embedding_batch[k, :output_len[k]]
                # embedding[:-1] because because embedding[-1] corresponded to EOF node
                embedding = torch.cat([embedding[:-1], shower.node_in_degree, shower.node_out_degree, shower.node_order], 1) # node order here
                # teacher forcing()
                showers_t.append(torch_geometric.data.Data(x=embedding,
                                                           edge_index=shower.adj_out,
                                                           edge_attr=shower.edge_weights).to(device))
                showers_x.append(shower.x[0])
            big_shower_t = torch_geometric.data.Batch.from_data_list(showers_t)
            showers_x = torch.cat(showers_x, dim=0)
            # GCN to recover shower features
            logits = features_nn.logits(big_shower_t, showers_x)
            ll_logits = -torch.mean(logits * big_shower_t.edge_attr)
            features = features_nn.generate(big_shower_t)
            ll_mse = loss_mse_edges(showers_x=showers_x, edge_index=big_shower_t.edge_index, features=features)

            # ll_smoothness, ll_IP_left, ll_IP_right = loss_edges(edge_index=big_shower_t.edge_index,
            #                                                     features=features,
            #                                                     add_dim=len(showers_batch) + 1)

        test_batch_energies = torch.tensor(np.random.uniform(low=1, high=5, size=(20, 1))).float().to(device)
        g = generate_graph(model=model,
                           edge_nn=edge_nn,
                           features_nn=features_nn,
                           max_prev_node=max_prev_node,
                           test_batch_energies=test_batch_energies,
                           device=device)

        ll_smoothness_gen, ll_IP_left_gen, ll_IP_right_gen = [], [], []
        for i in range(len(g)):
            l1, l2, l3 = loss_edges_over_predictions(edge_index=g[i].edge_index, predictions=g[i].x)
            ll_smoothness_gen.append(l1)
            ll_IP_left_gen.append(l2)
            ll_IP_right_gen.append(l3)
        ll_smoothness_gen = sum(ll_smoothness_gen) / (len(g) + 0.001)
        ll_IP_left_gen = sum(ll_IP_left_gen) / (len(g) + 0.001)
        ll_IP_right_gen = sum(ll_IP_right_gen) / (len(g) + 0.001)

        def num_to_float(x):
            if isinstance(x, torch.Tensor):
                return x.item()
            else:
                return x

        metrics['len_g'].append(len(g))
        metrics['average_shower_length'].append(sum([len(s.x) for s in g]) / (len(g) + 0.001))
        metrics['loss_average_bce'].append(ll_bce.item())
        metrics['loss_average_logits'].append(ll_logits.item())
        metrics['loss_smoothness'].append(ll_smoothness.item())
        metrics['loss_IP_left'].append(ll_IP_left.item())
        metrics['loss_IP_right'].append(ll_IP_right.item())
        metrics['loss_mse'].append(ll_mse.item())
        try:
            metrics['loss_smoothness_gen'].append(num_to_float(ll_smoothness_gen))
            metrics['loss_IP_left_gen'].append(num_to_float(ll_IP_left_gen))
            metrics['loss_IP_right_gen'].append(num_to_float(ll_IP_right_gen))
        except:
            pass

        loss = (
                loss_weights['ll_bce'] * ll_bce +
                loss_weights['ll_logits'] * ll_logits +
                loss_weights['ll_smoothness'] * ll_smoothness +
                loss_weights['ll_mse'] * ll_mse +
                loss_weights['ll_smoothness'] * ll_smoothness_gen +
                loss_weights['ll_IP_left'] * ll_IP_left_gen +
                loss_weights['ll_IP_right'] * ll_IP_right_gen
        )
        try:
            print({key: np.mean(metrics[key]) for key in metrics})
        except:
            pass
        if torch.isnan(loss):
            print("Loss is None!")
            continue
        loss.backward()
        optimizer.step()

    return metrics


def train(model, edge_nn, features_nn, dataset_loader, optimizer, device, experiment,
          max_prev_node, epochs=20000, signal_gen=True):
    experiment_key = experiment.get_key()
    for epoch in tqdm(range(epochs)):
        metrics = train_epoch(
            model,
            edge_nn,
            features_nn,
            dataset_loader,
            optimizer=optimizer,
            loss_weights={
                'll_bce': 1.,
                'll_logits': 1.,
                'll_smoothness': 0.0,
                'll_mse': 0.,
                'll_IP_left': 0.0,
                'll_IP_right': 0.0
            }, signal_gen=signal_gen, max_prev_node=max_prev_node, device=device)
        try:
            experiment.log_metrics({key: np.mean(metrics[key]) for key in metrics})
            print({key: np.mean(metrics[key]) for key in metrics})
        except:
            pass
        if (epoch + 1) % 1 == 0:
            PATH = './'
            torch.save(model.state_dict(), open(PATH + 'graph_nn_{}.pcl'.format(experiment_key), 'wb+'))
            torch.save(edge_nn.state_dict(), open(PATH + 'edge_nn_{}.pcl'.format(experiment_key), 'wb+'))
            torch.save(features_nn.state_dict(), open(PATH + 'features_nn_{}.pcl'.format(experiment_key), 'wb+'))


def shower_to_device(shower, device):
    return graphrnn_shower(adj=shower.adj.to(device),
                           x=shower.x.to(device),
                           diff_x=shower,
                           node_weights=shower.node_weights.to(device),
                           adj_out=shower.adj_out.to(device),
                           node_in_degree=shower.node_in_degree.to(device),
                           node_out_degree=shower.node_out_degree.to(device),
                           node_order=shower.node_order.to(device),
                           edge_weights=shower.edge_weights.to(device),
                           ele_p=shower.ele_p.to(device)
                           )


def get_freer_gpu():
    """
    Function to get the freest GPU available in the system
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

@click.command()
@click.option('--datafile', type=str, default="./data/showers_all_energies_1_5.pkl")
@click.option('--max_prev_node', type=int, default=12)
@click.option('--embedding_size', type=str, default=196)
@click.option('--edge_rnn_embedding_size', type=int, default=16)
@click.option('--embedding_size_gcn', type=int, default=4)
@click.option('--num_layers_gcn', type=int, default=3)
@click.option('--mixture_size', type=int, default=5)
@click.option('--lr', type=float, default=1e-4)
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
def main(
        datafile="./data/showers_all_energies_1_5.pkl",
        max_prev_node=12,
        embedding_size=196,
        edge_rnn_embedding_size=16,
        embedding_size_gcn=4,
        num_layers_gcn=4,
        mixture_size=12,
        lr=1e-4,
        work_space='schattengenie',
        project_name='shower_generation'
):
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(get_freer_gpu()))
    else:
        device = torch.device('cpu')
    print("Using device = {}".format(device))

    showers = pickle.load(open(datafile, "rb"))
    print(len(showers))
    experiment = Experiment(project_name=project_name, workspace=work_space)

    dataset_loader = torch.utils.data.DataLoader(
        TorchShowers(showers, max_prev_node=max_prev_node, device=device, q=0.05),
        pin_memory=True, batch_size=50, shuffle=True,
        num_workers=1, collate_fn=collate_fn
    )

    model = GraphRNN(input_size=max_prev_node,
                     embedding_size=embedding_size,  # embedding_size of linear layer
                     output_size=edge_rnn_embedding_size,
                     has_output=True,
                     hidden_size=embedding_size,
                     num_layers=4,
                     has_input=False).to(device)

    edge_nn = GraphRNN(input_size=1,
                       embedding_size=edge_rnn_embedding_size,
                       hidden_size=edge_rnn_embedding_size,
                       num_layers=4,
                       has_input=True,
                       has_output=True,
                       output_size=1).to(device)

    features_nn = FeaturesGCN(dim_in=embedding_size + 3,  # node order here
                              embedding_size=embedding_size_gcn,
                              num_layers_gcn=num_layers_gcn,
                              num_layers_dense=0,
                              mixture_size=mixture_size,
                              dim_out=5).to(device=device)

    parameters = list(features_nn.parameters()) + list(model.parameters()) + list(edge_nn.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-4)

    train(
        model=model,
        edge_nn=edge_nn,
        features_nn=features_nn,
        dataset_loader=dataset_loader,
        optimizer=optimizer,
        device=device,
        experiment=experiment,
        max_prev_node=max_prev_node
    )


if __name__ == '__main__':
    main()
