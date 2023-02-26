"""
    
"""

import os

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.loader import DataLoader

class GraphCheb(torch.nn.Module):
    """
    graphcheb classifier
    """

    def __init__(self,
                 num_node_features: int,
                 hidden_channels: int,
                 K: int,
                 layers_nr: int,
                 num_classes: int,
                 ):
        """
        Init

        :param num_node_features: Number of node features 
        :param hidden_channels: Number of neurons in each layer
        :param K ---
        :param layers_nr: Number of layers of the GNN
        :param num_classes: Number of output classes
        """

        super(GraphCheb, self).__init__()

        #torch.manual_seed(12345)

        self.hidden_channels = hidden_channels
        self.layers_nr = layers_nr

        self.cheb_layers_list = []      # [1.] All layers in a list ----------------------------------------------------

        # [2.] Input layer ---------------------------------------------------------------------------------------------
        self.cheb_layers_list.append(ChebConv(num_node_features, self.hidden_channels, K))

        # [3.] Intermediate layers -------------------------------------------------------------------------------------
        for intermediate_layer in range(self.layers_nr - 1):
            print(f"Intermediate Layer: {intermediate_layer}")
            self.cheb_layers_list.append(ChebConv(self.hidden_channels, self.hidden_channels, K))

        self.cheb_modules = nn.ModuleList(self.cheb_layers_list)

        # [4.] Last layer ----------------------------------------------------------------------------------------------
        self.lin = Linear(self.hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Forward

        :param x:
        :param edge_index:
        :param batch:
        :param edge_weight:
        :return:
        """

        # [1.] Obtain node embeddings ----------------------------------------------------------------------------------
        cheb_modules_len = len(self.cheb_modules)
        for cheb_module_idx in range(cheb_modules_len):

            cheb_module = self.cheb_modules[cheb_module_idx]

            if cheb_module_idx < cheb_modules_len - 1:

                x = cheb_module(x, edge_index, edge_weight)
                x = x.relu()
            else:
                x = cheb_module(x, edge_index, edge_weight)

        # [2.] Readout layer -------------------------------------------------------------------------------------------
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # [3.] Apply a final classifier --------------------------------------------------------------------------------
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x