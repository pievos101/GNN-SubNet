import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import copy
from matplotlib import pyplot as plt
import networkx as nx
from tqdm import tqdm

from dataset import generate, convert_to_s2vgraph
from gnn import MUTAG_Classifier
from pg_explainer import PGExplainer
from graphcnn import GraphCNN
from gnn_training_utils import pass_data_iteratively

nodes_per_graph_nr = 30
graph = nx.generators.random_graphs.barabasi_albert_graph(nodes_per_graph_nr, 2)
# Get edges of graph -----------------------------------------------------------------------------------------------
edges = list(graph.edges())

#Select nodes for label calculation --------------------------------------------------------------------------------
edge_idx = np.random.randint(len(edges))

node_indices = [edges[edge_idx][0], edges[edge_idx][1]]
sigma = 0.1
no_of_features = 1

dataset, path = generate(500, nodes_per_graph_nr, sigma, graph, node_indices, no_of_features)

#path = 'graphs_3_11'
#dataset = load_syn_dataset(path, type_of_feat='float')



print('--------DATASET LOADED-------------')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=123)
s2v_train_dataset = convert_to_s2vgraph(train_dataset)
s2v_test_dataset = convert_to_s2vgraph(test_dataset)
#train_dataloader = DataLoader(
#    train_dataset,
#    batch_size=64,
#    shuffle=True)

epoch_nr = 50

input_dim = no_of_features
n_classes = 2

model = GraphCNN(5, 2, input_dim, 32, n_classes, 0.5, True, 'sum1', 'sum', 0)
opt = torch.optim.Adam(model.parameters(), lr = 0.01)
model.train()
min_loss = 50
best_model = GraphCNN(5, 3, input_dim, 32, n_classes, 0.5, True, 'sum1', 'sum', 0)
min_val_loss = 150
n_epochs_stop = 5
epochs_no_improve = 0
steps_per_epoch = 35

for epoch in range(epoch_nr):
    model.train()
    pbar = tqdm(range(steps_per_epoch), unit='batch')
    epoch_loss = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(s2v_train_dataset))[:32]

        batch_graph = [s2v_train_dataset[idx] for idx in selected_idx]
        logits = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph])
        loss = nn.CrossEntropyLoss()(logits,labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= steps_per_epoch
    model.eval()
    output = pass_data_iteratively(model, s2v_train_dataset)
    predicted_class = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in s2v_train_dataset])
    correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
    acc_train = correct / float(len(s2v_train_dataset))
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    print(f"Train Acc {acc_train:.4f}")


    pbar.set_description('epoch: %d' % (epoch))
    val_loss = 0
    output = pass_data_iteratively(model, s2v_test_dataset)

    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
    loss = nn.CrossEntropyLoss()(output,labels)
    val_loss += loss

    print('Epoch {}, val_loss {:.4f}'.format(epoch, val_loss))
    if val_loss < min_val_loss:
        print(f"Saving best model with loss {val_loss}")
        best_model = copy.deepcopy(model)
        epochs_no_improve = 0
        min_val_loss = val_loss

    else:
        epochs_no_improve += 1
        # Check early stopping condition
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            model.load_state_dict(best_model.state_dict())
            break

confusion_array = []
true_class_array = []
predicted_class_array = []
model.eval()
correct = 0
true_class_array = []
predicted_class_array = []

test_loss = 0

model.load_state_dict(best_model.state_dict())

output = pass_data_iteratively(model, s2v_test_dataset)
predicted_class = output.max(1, keepdim=True)[1]
labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
acc_test = correct / float(len(s2v_test_dataset))

loss = nn.CrossEntropyLoss()(output,labels)
test_loss = loss

predicted_class_array = np.append(predicted_class_array, predicted_class)
true_class_array = np.append(true_class_array, labels)

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

'''
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
'''

train_graphs = s2v_train_dataset
z = model(train_graphs, get_embedding=True)
exp = PGExplainer(model, 32, task="graph", log=True)
exp.train_explainer_s2v(train_dataset, z, train_graphs, None)
test_graphs = s2v_test_dataset
z = model(test_graphs, get_embedding=True)

#edge_mask = exp.explain_s2v(test_dataset, z)
test_graphs = Batch.from_data_list(test_dataset)
edge_mask = exp.explain(test_graphs, z)

em = np.reshape(edge_mask, (len(test_dataset), -1))

Path(f"{path}/pg_results").mkdir(parents=True, exist_ok=True)
#np.savetxt(f'{path}/pg_results/pg_edge_masks.csv', edge_mask, delimiter=',', fmt='%.3f')
np.savetxt(f'{path}/pg_results/pg_edge_masks.csv', em, delimiter=',', fmt='%.3f')



    