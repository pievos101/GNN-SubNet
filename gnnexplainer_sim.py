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
from torch_geometric.nn import conv
from tqdm import tqdm


from gnn_training_utils import pass_data_iteratively
from dataset import generate, save_results, load_syn_dataset, convert_to_s2vgraph
from gnn_explainer import GNNExplainer as gnnexp
from gnn_explainer import GNNExplainer
from gnn_training_utils import check_if_graph_is_connected
from community_detection import find_communities
from edge_importance import calc_edge_importance

from graphcnn import GraphCNN

nodes_per_graph_nr = 30
graph = nx.generators.random_graphs.barabasi_albert_graph(nodes_per_graph_nr, 1)
# Get edges of graph -----------------------------------------------------------------------------------------------
edges = list(graph.edges())
#Select nodes for label calculation --------------------------------------------------------------------------------
edge_idx = np.random.randint(len(edges))

node_indices = [edges[edge_idx][0], edges[edge_idx][1]]
sigma = 1
no_of_features = 1
dataset, path = generate(500, nodes_per_graph_nr, sigma, graph, node_indices, no_of_features)
dataset = convert_to_s2vgraph(dataset)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True)

epoch_nr = 10

input_dim = no_of_features
n_classes = 2
model = GraphCNN(5, 3, input_dim, 32, n_classes, 0.5, True, 'sum1', 'sum', 0)
opt = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = ReduceLROnPlateau(opt, 'min')

model.train()

for epoch in range(epoch_nr):
    model.train()
    pbar = tqdm(range(50), unit='batch')
    epoch_loss = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_dataset))[:32]

        batch_graph = [train_dataset[idx] for idx in selected_idx]
        logits = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph])
        loss = F.cross_entropy(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.detach().item()
    epoch_loss /= 50
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    pbar.set_description('epoch: %d' % (epoch))

confusion_array = []
true_class_array = []
predicted_class_array = []
model.eval()
correct = 0
true_class_array = []
predicted_class_array = []

test_loss = 0

output = pass_data_iteratively(model, test_dataset)
predicted_class = output.max(1, keepdim=True)[1]
labels = torch.LongTensor([graph.label for graph in test_dataset])
correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
acc_test = correct / float(len(test_dataset))

#output = best_model(data, torch.tensor(batch))
#predicted_class = output.max(dim=1)[1]
#true_class = data.y.item()
loss = F.cross_entropy(output, labels)
test_loss = loss

predicted_class_array = np.append(predicted_class_array, predicted_class)
true_class_array = np.append(true_class_array, labels)

#correct += predicted_class.eq(data.y).sum().item()

#test_loss /= len(test_dataset)
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

print("")
print("Run the Explainer ...")

no_of_runs = 10
lamda = 0.85 # not used anymore
ems = []
for idx in range(no_of_runs):
    print(f'Explainer::Iteration {idx+1} of {no_of_runs}') 
    exp = GNNExplainer(model, epochs=300)
    em = exp.explain_graph_modified_s2v(dataset, lamda)
    Path(f"{path}/{sigma}/modified_gnn").mkdir(parents=True, exist_ok=True)
    gnn_feature_masks = np.reshape(em, (len(em), -1))
    np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
    gnn_edge_masks = calc_edge_importance(gnn_feature_masks, dataset[0].edge_mat)
    np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
    ems.append(gnn_edge_masks.sigmoid().numpy())
    
ems = np.array(ems)
mean_em = ems.mean(0)

np.savetxt(f"{path}/edge_masks.csv", mean_em, delimiter=',', fmt='%.5f')
avg_mask, coms = find_communities(f"{path}/dataset/graph0_edges.txt", f"{path}/edge_masks.csv")
print(avg_mask, coms)

