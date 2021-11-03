import numpy as np
from scipy.sparse.extract import find
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn.modules import conv
from torch_geometric import data
from torch_geometric.data import DataLoader, Batch
from pathlib import Path
import copy
from tqdm import tqdm

from torch_geometric.data.data import Data

from gnn_training_utils import check_if_graph_is_connected, pass_data_iteratively
from dataset import generate, load_KIRC_dataset, convert_to_s2vgraph
from gnn import MUTAG_Classifier
from pg_explainer import PGExplainer
from gnn_explainer import GNNExplainer
from graphcnn import GraphCNN


from community_detection import find_communities
from Edge_Importance import calc_edge_importance


LOC = "/home/bastian/LinkedOmics/KIRC"

dataset, col_pairs, row_pairs = load_KIRC_dataset(f'{LOC}/KIDNEY_PPI.txt', 
                                [f'{LOC}/KIDNEY_Methy_FEATURES.txt', f'{LOC}/KIDNEY_mRNA_FEATURES.txt'], 
                                 f'{LOC}/KIDNEY_SURVIVAL.txt')



#dataset, col_pairs, row_pairs = load_KIRC_dataset("/home/bastian/LinkedOmics/KIRC/KIDNEY_PPI.txt", 
#                                ["/home/bastian/LinkedOmics/KIRC/KIDNEY_Methy_FEATURES.txt", "/home/bastian/LinkedOmics/KIRC/KIDNEY_mRNA_FEATURES.txt"], 
#                                 "/home/bastian/LinkedOmics/KIRC/KIDNEY_SURVIVAL.txt")

#dataset, col_pairs, row_pairs = load_KIRC_dataset("KIRC-OV/KIDNEY_OV_PPI.txt", 
#                                ["KIRC-OV/KIDNEY_OV_Methy_FEATURES.txt", "KIRC-OV/KIDNEY_OV_mRNA_FEATURES.txt"], "KIRC-OV/KIDNEY_OV_TARGET.txt")

print("Graph is connected", check_if_graph_is_connected(dataset[0].edge_index))

count = 0
for item in dataset:
    count += item.y.item()

use_weights = False
weight = torch.tensor([count/len(dataset), 1-count/len(dataset)])
print(count/len(dataset), 1-count/len(dataset))

model_path = 'kirc_model.pth'
no_of_features = dataset[0].x.shape[1]
nodes_per_graph_nr = dataset[0].x.shape[0]

load_model = False

print(len(dataset), len(dataset)*0.2)
#s2v_dataset = convert_to_s2vgraph(dataset)
print('--------DATASET LOADED-------------')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=123)
s2v_train_dataset = convert_to_s2vgraph(train_dataset)
s2v_test_dataset = convert_to_s2vgraph(test_dataset)
#s2v_train_dataset, s2v_test_dataset = train_test_split(s2v_dataset, test_size=0.2, random_state=123)

epoch_nr = 150

input_dim = no_of_features
n_classes = 2

model = GraphCNN(5, 2, input_dim, 32, n_classes, 0.5, True, 'sum1', 'sum', 0)
opt = torch.optim.Adam(model.parameters(), lr = 0.01)

if load_model:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    opt = checkpoint['optimizer']

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
        if use_weights:
            loss = nn.CrossEntropyLoss(weight=weight)(logits,labels)
        else:
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
    if use_weights:
            loss = nn.CrossEntropyLoss(weight=weight)(output,labels)
    else:
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

if use_weights:
    loss = nn.CrossEntropyLoss(weight=weight)(output,labels)
else:
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

checkpoint = {
    'state_dict': best_model.state_dict(),
    'optimizer': opt.state_dict()
}
torch.save(checkpoint, model_path)

model.train()

# add tweak explainer in comment

###############################################################
# Thats the PGExplainer Call ##################################
###############################################################
#train_graphs = s2v_train_dataset
#print(len(train_graphs))
#z = model(train_graphs, get_embedding=True)
#exp = PGExplainer(model, 32, task="graph", log=True)
#exp.train_explainer_s2v(train_dataset, z, train_graphs, None)
#test_graphs = s2v_test_dataset
#z = model(test_graphs, get_embedding=True)
#edge_mask = exp.explain_s2v(test_dataset, z)
#print(edge_mask.shape, train_dataset[0].edge_index.shape)

#em = np.reshape(edge_mask, (len(s2v_test_dataset), -1))
#np.savetxt('KIRC/edge_masks.txt', edge_mask, fmt='%.3f')

#avg_mask, coms = find_communities("KIRC/edge_index.txt", "KIRC/edge_masks.txt")
#print(avg_mask, coms)
###############################################################
###############################################################
###############################################################

print("")
print("Run the Explainer ...")

no_of_runs = 3
lamda = 0.8 # not used anymore
ems = []
#@FIXME set output to the location of the input data
for idx in range(no_of_runs):
    print(f'Explainer::Iteration {idx+1} of {no_of_runs}') 
    exp = GNNExplainer(model, epochs=300)
    em = exp.explain_graph_modified_s2v(s2v_test_dataset, lamda)
    #Path(f"{path}/{sigma}/modified_gnn").mkdir(parents=True, exist_ok=True)
    gnn_feature_masks = np.reshape(em, (len(em), -1))
    np.savetxt(f'{LOC}/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
    #np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
    gnn_edge_masks = calc_edge_importance(gnn_feature_masks, dataset[0].edge_index)
    np.savetxt(f'{LOC}/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
    #np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
    ems.append(gnn_edge_masks.sigmoid().numpy())
    
ems = np.array(ems)
mean_em = ems.mean(0)

np.savetxt(f'{LOC}/edge_masks.csv', mean_em, delimiter=',', fmt='%.5f')
avg_mask, coms = find_communities(f'{LOC}/edge_index.txt', f'{LOC}/edge_masks.csv')
#print(avg_mask, coms)
