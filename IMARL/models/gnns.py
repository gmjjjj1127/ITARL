import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T
from torch_geometric.nn import  GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool



class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, L, dropout, use_bn=False, task='node', pool='mean'):
        super(GCN, self).__init__()

        self.L = L
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        #self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))
        for _ in range(self.L - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            #self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.use_bn = use_bn

        self.task = task
        self.pool = pool
        if task == 'graph':
            self.graph_linear = nn.Linear(out_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        if self.task == 'graph':
            if self.pool == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pool == 'max':
                x = global_max_pool(x, batch)
            elif self.pool == 'add':
                x = global_add_pool(x, batch)
            x = self.graph_linear(x)
            # print(x.T)
        return x
