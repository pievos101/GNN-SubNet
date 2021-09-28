import numpy as np
from scipy.sparse.extract import find
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader, Batch
from pathlib import Path
import copy

from dataset import generate, load_KIRC_dataset, generate_confusion
from gnn import MUTAG_Classifier
from pg_explainer import PGExplainer
from gnn_explainer import GNNExplainer

from community_detection import find_communities

dataset, col_pairs, row_pairs = load_KIRC_dataset("KIRC/KIDNEY_PPI.txt", 
                                ["/KIRC/KIDNEY_Methy_FEATURES.txt", 
                                 "/KIRC/KIDNEY_mRNA_FEATURES.txt"], 
                                 "/KIRC/KIDNEY_SURVIVAL.txt")
print('--------DATASET LOADED-------------')
model_path = 'kirc_model.pth'
no_of_features = dataset[0].x.shape[1]
nodes_per_graph_nr = dataset[0].x.shape[0]
load_model = False

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True)

epoch_nr = 50

input_dim = no_of_features
n_classes = 2

model = MUTAG_Classifier(input_dim, n_classes)
opt = torch.optim.Adam(model.parameters(), lr = 0.01)

if load_model:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])

model.train()
min_loss = 50
best_model = MUTAG_Classifier(input_dim, n_classes)
min_val_loss = 50
n_epochs_stop = 5
epochs_no_improve = 0

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
    #scheduler.step(loss)
    epoch_loss /= graph_idx
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))

    val_loss = 0
    for data in test_dataset:
        batch = []
        for i in range(nodes_per_graph_nr):
            batch.append(0)

        output = model(data, torch.tensor(batch))
        loss = F.nll_loss(output, torch.tensor([data.y]))
        val_loss += loss

    val_loss /= len(test_dataset)
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
            print('Early stopping!' )
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

for data in test_dataset:
    batch = []
    for i in range(nodes_per_graph_nr):
        batch.append(0)

    output = best_model(data, torch.tensor(batch))
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

checkpoint = {
    'state_dict': best_model.state_dict(),
    'optimizer': opt.state_dict()
}
torch.save(checkpoint, model_path)

model.train()

train_graphs = Batch.from_data_list(train_dataset)
z = model(train_graphs, train_graphs.batch,
        get_embedding=True)
exp = PGExplainer(model, 32, task="graph", log=True)
exp.train_explainer(train_graphs, z, None,
                    train_graphs.batch)
test_graphs = Batch.from_data_list(test_dataset)
z = model(test_graphs, batch=test_graphs.batch,
        get_embedding=True)
edge_mask = exp.explain(test_graphs, z)

em = np.reshape(edge_mask, (len(test_dataset), -1))
np.savetxt('KIRC/edge_masks.txt', em, fmt='%.3f')

avg_mask, coms = find_communities("KIRC/edge_index.txt", "KIRC/edge_masks.txt")
print(avg_mask, coms)
