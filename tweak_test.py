import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import copy
from matplotlib import pyplot as plt
import networkx as nx

from dataset import generate, save_results, generate_confusion, load_syn_dataset
from gnn import MUTAG_Classifier
from gnn_explainer import GNNExplainer as gnnexp
from pg_explainer import PGExplainer
from gnn_explainer import GNNExplainer
from gnn_training_utils import check_if_graph_is_connected

nodes_per_graph_nr = 20
graph = nx.generators.random_graphs.barabasi_albert_graph(nodes_per_graph_nr, 1)
# Get edges of graph -----------------------------------------------------------------------------------------------
edges = list(graph.edges())
#Select nodes for label calculation --------------------------------------------------------------------------------
edge_idx = np.random.randint(len(edges))

node_indices = [edges[edge_idx][0], edges[edge_idx][1]]
sigma = 0.5
no_of_features = 2
dataset, path = generate(500, nodes_per_graph_nr, sigma, graph, node_indices, no_of_features)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True)

epoch_nr = 50

input_dim = no_of_features
n_classes = 2
model = MUTAG_Classifier(input_dim, n_classes)
opt = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = ReduceLROnPlateau(opt, 'min')

model.train()

for epoch in range(epoch_nr):
    epoch_loss = 0
    graph_idx = 0
    for data in train_dataloader:
        batch = []
        for i in range(data.y.size(0)):
            for j in range(nodes_per_graph_nr):
                batch.append(i)
        logits = model(data, torch.tensor(batch))
        loss = F.nll_loss(logits, data.y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.detach().item()
        graph_idx += 1
    scheduler.step(loss)
    epoch_loss /= graph_idx

confusion_array = []
true_class_array = []
predicted_class_array = []
model.eval()
correct = 0
true_class_array = []
predicted_class_array = []

test_loss = 0

for data in test_dataset:
    batch = []
    for i in range(nodes_per_graph_nr):
        batch.append(0)

    output = model(data, torch.tensor(batch))
    predicted_class = output.max(dim=1)[1]
    true_class = data.y.item()
    loss = F.nll_loss(output, torch.tensor([data.y]))
    test_loss += loss
    confusion_array.append(generate_confusion(true_class, predicted_class))

    predicted_class_array = np.append(predicted_class_array, predicted_class)
    true_class_array = np.append(true_class_array, true_class)

    correct += predicted_class.eq(data.y).sum().item()

test_loss /= len(test_dataset)
confusion_matrix_gnn = confusion_matrix(true_class_array, predicted_class_array)
print("\nConfusion matrix:\n")
print(confusion_matrix_gnn)


counter = 0
for it, i in zip(predicted_class_array, range(len(predicted_class_array))):
    if it == true_class_array[i]:
        counter += 1

accuracy = counter/len(true_class_array) * 100 
print("Accuracy: {}%".format(accuracy))
print("Test loss {}".format(test_loss))

model.train()

no_of_runs = 20
lamda = 0.85
for idx in range(no_of_runs):
    exp = GNNExplainer(model, epochs=600)
    em = exp.explain_graph_modified(dataset, lamda)
    Path(f"{path}/{sigma}/modified_gnn").mkdir(parents=True, exist_ok=True)
    gnn_edge_masks = np.reshape(em, (len(em), -1))
    np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_edge_masks{idx}.csv', gnn_edge_masks, delimiter=',', fmt='%.3f')
