import torch
from torch import nn
import torch_geometric.transforms as T
import torch_cluster
import torch_geometric
from torch_geometric.nn import NNConv, GCNConv, GraphConv
from torch_geometric.nn import PointConv, EdgeConv, SplineConv
from gmn_net import GaussianMixtureNetwork


class FeaturesGCN(nn.Module):
    def __init__(self, dim_in, embedding_size=16,
                 num_layers_gcn=3,
                 num_layers_dense=4,
                 dim_out=5,
                 mixture_size=5,
                 agg='mean'):
        super().__init__()
        self.wconv_in = EdgeConv(nn.Sequential(
            nn.Linear(dim_in * 2, embedding_size),
            nn.Dropout(p=0.1),
            nn.Tanh()), agg)
        self.layers_gcn = nn.ModuleList(modules=[EdgeConv(
            nn.Sequential(
                nn.Linear(embedding_size * 2, embedding_size),
                nn.Dropout(p=0.1),
                nn.Tanh()
            ),
            agg) for _ in range(num_layers_gcn)])
        self.layers_dense = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.Tanh()
            )
            for _ in range(num_layers_dense)
        ])
        self.out = GaussianMixtureNetwork(embedding_size * 2, mixture_size, targets=dim_out)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.wconv_in(x=x, edge_index=edge_index)
        for l in self.layers_gcn:
            x = l(x=x, edge_index=edge_index)
        for l in self.layers_dense:
            x = l(x)
        x = torch.cat([
            torch.index_select(x, dim=0, index=edge_index[0]),
            torch.index_select(x, dim=0, index=edge_index[1])
        ], dim=1)
        return x, edge_index

    def get_distr(self, data):
        x, edge_index = self.forward(data)
        return self.out(x)

    def logits(self, data, target):

        x, edge_index = self.forward(data)
        logits = self.out.logits(x, torch.index_select(target, dim=0, index=edge_index[1]) - torch.index_select(target, dim=0, index=edge_index[0]))
        return logits

    def generate(self, data):
        x, edge_index = self.forward(data)
        return self.out.generate(x)


    def generate_mll(self, data):
        x, edge_index = self.forward(data)
        return self.out.generate_mll(x)
