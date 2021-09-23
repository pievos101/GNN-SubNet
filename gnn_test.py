from dataset import generate, generate_confusion
from gnn import MUTAG_Classifier
from gnn_explainer import GNNExplainer
import networkx as nx 
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from matplotlib import pyplot as plt

nodes_per_graph_nr = 20
path = 'graphs_1_4'
#dataset = load_dataset(path, type_of_feat="float")
graph = nx.generators.random_graphs.barabasi_albert_graph(nodes_per_graph_nr, 1)
# Get edges of graph -----------------------------------------------------------------------------------------------
edges = list(graph.edges())

#Select nodes for label calculation --------------------------------------------------------------------------------
edge_idx = np.random.randint(len(edges))

node_indices = [edges[edge_idx][0], edges[edge_idx][1]]
sigma=0
dataset, path = generate(500, nodes_per_graph_nr, sigma, graph, node_indices)

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True)

epoch_nr = 50

input_dim = 1
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

graph = test_dataset[0]
explainer = GNNExplainer(model, epochs=500)
nfm, em, logits_init, logits_post = explainer.explain_graph(graph, torch.zeros(nodes_per_graph_nr))
em = em.detach().numpy()

Path(f"{path}/gnn_explainer").mkdir(parents=True, exist_ok=True)
np.savetxt(f'{path}/gnn_explainer/gnn_edge_masks.csv', em, delimiter=',', fmt='%.3f')

    
