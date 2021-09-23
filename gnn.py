"""
    Graph Neural Network

    :author: Afan Secic (main developer), Anna Saranti (corrections and refactoring)
    :copyright: © 2020 HCI-KDD (ex-AI) group
    :date: 2020-12-01
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool, GINConv

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d


class MUTAG_Classifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MUTAG_Classifier, self).__init__()
        #Architecture taken from example for MUTAG dataset on graph classification
        dim = 32

        self.conv1 = GINConv(
            Sequential(Linear(input_dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, num_classes)

    def forward(self, data, batch, get_embedding=False):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        if get_embedding:
            return x
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

