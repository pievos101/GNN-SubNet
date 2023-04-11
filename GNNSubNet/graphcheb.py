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

def test_model_acc(loader, model):
    """
    Returns test accuracy.
    """
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def test_model(loader, model, criterion=None):
    """
    Tests model, returns test loss, test accuracy, list of predicted labels, list of true labels.
    """
    correct = 0
    running_loss = 0.0
    steps = 0
    predicted_labels = [] # list to keep predicted labels
    true_labels = []
    model.eval()
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        if criterion:
            running_loss += criterion(out, data.y)
        else:
            running_loss = 0.0
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        steps += 1
        predicted_labels.extend(list(pred.detach().cpu().numpy()))
        true_labels.extend(list(data.y.detach().cpu().numpy()))
    return running_loss / steps, correct / len(loader.dataset), true_labels, predicted_labels  # Derive ratio of correct predictions.

class ChebConvNet(nn.Module):
    """
    ChebNet with one convolutional layer, ReLU nonlinearity, and fixed number of nodes for every single data point input.
    The purpose of this model is graph signal classification.
    """
    def __init__(self, input_channels, n_features, n_channels, n_classes, K, n_layers):
        super(ChebConvNet, self).__init__()
        self.cheb_graph_convs = nn.ModuleList()
        self.input_channels = input_channels # number of channels in the input
        self.n_classes = n_classes
        self.n_features = n_features # number of nodes in a graph
        self.K = K # degree of the polynomial
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.cheb_graph_convs.append(ChebConv(in_channels=self.input_channels, out_channels=self.n_channels, K=self.K))
        for k in range(1, self.n_layers - 1):
            self.cheb_graph_convs.append(ChebConv(in_channels=self.n_channels, out_channels=self.n_channels, K=self.K))
        # self.cheb_graph_convs.append(ChebGraphConv(K_order, n_hid, n_class, enable_bias))
        self.relu = nn.ReLU()
        self.lin = nn.Linear(self.n_features * self.n_channels, n_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        for k in range(self.n_layers):
            x = self.cheb_graph_convs[k](x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch, lambda_max=2)
            x = self.relu(x)
            # print("After ChebNet")
            # print(x.size())

        batch_size = x.size()[0] / self.n_features
        x = torch.reshape(x, (int(batch_size), self.n_features * self.n_channels))
        x = self.lin(x)
        return x

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
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        return x