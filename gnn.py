"""
    Graph Neural Network

    :author: Afan Secic (main developer), Anna Saranti (corrections and refactoring)
    :copyright: © 2020 HCI-KDD (ex-AI) group
    :date: 2020-12-01
"""

from torch_geometric.utils import get_laplacian
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool, GINConv

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn.glob.glob import global_max_pool, global_mean_pool

class Net(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        dim = 32
        self.conv1 = GCNConv(input_dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.conv3 = GCNConv(dim, dim)
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, dim // 2)
        self.lin3 = torch.nn.Linear(dim // 2, num_classes)
    
    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        #x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        #x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, enhance=False):
        super(MLP, self).__init__()

        self.enhance = enhance

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        if enhance:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        if self.enhance:
            x = self.bn1(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.enhance:
            x = self.bn2(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc3(x)

        return x


class MEWISPool(nn.Module):
    def __init__(self, hidden_dim):
        super(MEWISPool, self).__init__()

        self.gc1 = GINConv(MLP(1, hidden_dim, hidden_dim))
        self.gc2 = GINConv(MLP(hidden_dim, hidden_dim, hidden_dim))
        self.gc3 = GINConv(MLP(hidden_dim, hidden_dim, 1))

    def forward(self, x, edge_index, batch):
        # computing the graph laplacian and adjacency matrix
        batch_nodes = batch.size(0)
        if edge_index.size(1) != 0:
            L_indices, L_values = get_laplacian(edge_index)
            L = torch.sparse.FloatTensor(L_indices, L_values, torch.Size([batch_nodes, batch_nodes]))
            A = torch.diag(torch.diag(L.to_dense())) - L.to_dense()

            # entropy computation
            entropies = self.compute_entropy(x, L, A, batch)  # Eq. (8)
        else:
            A = torch.zeros([batch_nodes, batch_nodes])
            norm = torch.norm(x, dim=1).unsqueeze(-1)
            entropies = norm / norm

        # graph convolution and probability scores
        probabilities = self.gc1(entropies, edge_index)
        probabilities = self.gc2(probabilities, edge_index)
        probabilities = self.gc3(probabilities, edge_index)
        probabilities = torch.sigmoid(probabilities)

        # conditional expectation; Algorithm 1
        gamma = entropies.sum()
        loss = self.loss_fn(entropies, probabilities, A, gamma)  # Eq. (9)

        mewis = self.conditional_expectation(entropies, probabilities, A, loss, gamma)

        # graph reconstruction; Eq. (10)
        x_pooled, adj_pooled = self.graph_reconstruction(mewis, x, A)
        edge_index_pooled, batch_pooled = self.to_edge_index(adj_pooled, mewis, batch)

        return x_pooled, edge_index_pooled, batch_pooled, loss, mewis

    @staticmethod
    def compute_entropy(x, L, A, batch):
        # computing local variations; Eq. (5)
        V = x * torch.matmul(L, x) - x * torch.matmul(A, x) + torch.matmul(A, x * x)
        V = torch.norm(V, dim=1)

        # computing the probability distributions based on the local variations; Eq. (7)
        P = torch.cat([torch.softmax(V[batch == i], dim=0) for i in torch.unique(batch)])
        P[P == 0.] += 1
        # computing the entropies; Eq. (8)
        H = -P * torch.log(P)

        return H.unsqueeze(-1)

    @staticmethod
    def loss_fn(entropies, probabilities, A, gamma):
        term1 = -torch.matmul(entropies.t(), probabilities)[0, 0]

        term2 = torch.matmul(torch.matmul(probabilities.t(), A), probabilities).sum()

        return gamma + term1 + term2

    def conditional_expectation(self, entropies, probabilities, A, threshold, gamma):
        sorted_probabilities = torch.sort(probabilities, descending=True, dim=0)

        dummy_probabilities = probabilities.detach().clone()
        selected = set()
        rejected = set()

        for i in range(sorted_probabilities.values.size(0)):
            node_index = sorted_probabilities.indices[i].item()
            neighbors = torch.where(A[node_index] == 1)[0]
            if len(neighbors) == 0:
                selected.add(node_index)
                continue
            if node_index not in rejected and node_index not in selected:
                s = dummy_probabilities.clone()
                s[node_index] = 1
                s[neighbors] = 0

                loss = self.loss_fn(entropies, s, A, gamma)

                if loss <= threshold:
                    selected.add(node_index)
                    for n in neighbors.tolist():
                        rejected.add(n)

                    dummy_probabilities[node_index] = 1
                    dummy_probabilities[neighbors] = 0

        mewis = list(selected)
        mewis = sorted(mewis)

        return mewis

    @staticmethod
    def graph_reconstruction(mewis, x, A):
        x_pooled = x[mewis]

        A2 = torch.matmul(A, A)
        A3 = torch.matmul(A2, A)

        A2 = A2[mewis][:, mewis]
        A3 = A3[mewis][:, mewis]

        I = torch.eye(len(mewis))
        one = torch.ones([len(mewis), len(mewis)])

        adj_pooled = (one - I) * torch.clamp(A2 + A3, min=0, max=1)

        return x_pooled, adj_pooled

    @staticmethod
    def to_edge_index(adj_pooled, mewis, batch):
        row1, row2 = torch.where(adj_pooled > 0)
        edge_index_pooled = torch.cat([row1.unsqueeze(0), row2.unsqueeze(0)], dim=0)
        batch_pooled = batch[mewis]

        return edge_index_pooled, batch_pooled


class Net3(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net3, self).__init__()

        hidden_dim = 32

        self.gc1 = GINConv(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc2 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool1 = MEWISPool(hidden_dim=hidden_dim)
        self.gc3 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc4 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool2 = MEWISPool(hidden_dim=hidden_dim)
        self.gc5 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index

        x = self.gc1(x, edge_index)
        x = torch.relu(x)

        x = self.gc2(x, edge_index)
        x = torch.relu(x)

        x_pooled1, edge_index_pooled1, batch_pooled1, loss1, mewis1 = self.pool1(x, edge_index, batch)

        x_pooled1 = self.gc3(x_pooled1, edge_index_pooled1)
        x_pooled1 = torch.relu(x_pooled1)

        x_pooled1 = self.gc4(x_pooled1, edge_index_pooled1)
        x_pooled1 = torch.relu(x_pooled1)

        x_pooled2, edge_index_pooled2, batch_pooled2, loss2, mewis2 = self.pool2(x_pooled1, edge_index_pooled1,
                                                                                 batch_pooled1)

        x_pooled2 = self.gc5(x_pooled2, edge_index_pooled2)
        x_pooled2 = torch.relu(x_pooled2)

        readout = torch.cat([x_pooled2[batch_pooled2 == i].mean(0).unsqueeze(0) for i in torch.unique(batch_pooled2)],
                            dim=0)

        out = self.fc1(readout)
        out = torch.relu(out)
        out = self.fc2(out)

        return torch.log_softmax(out, dim=-1)

class Net4(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net4, self).__init__()
        hidden_dim = 32
        self.gc1 = GINConv(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool1 = MEWISPool(hidden_dim=hidden_dim)
        self.gc2 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool2 = MEWISPool(hidden_dim=hidden_dim)
        self.gc3 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = torch.relu(x)

        x_pooled1, edge_index_pooled1, batch_pooled1, loss1, mewis = self.pool1(x, edge_index, batch)

        x_pooled1 = self.gc2(x_pooled1, edge_index_pooled1)
        x_pooled1 = torch.relu(x_pooled1)

        x_pooled2, edge_index_pooled2, batch_pooled2, loss2, mewis = self.pool2(x_pooled1, edge_index_pooled1,
                                                                                batch_pooled1)

        x_pooled2 = self.gc3(x_pooled2, edge_index_pooled2)

        readout = torch.cat([x_pooled2[batch_pooled2 == i].mean(0).unsqueeze(0) for i in torch.unique(batch_pooled2)],
                            dim=0)

        out = self.fc1(readout)
        out = torch.relu(out)
        out = self.fc2(out)

        return torch.log_softmax(out, dim=-1)