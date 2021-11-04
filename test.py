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

from dataset import generate, save_results, generate_confusion
from gnn import MUTAG_Classifier, Classifier
from gnn_explainer import GNNExplainer as gnnexp
from pg_explainer import PGExplainer
from gnn_explainer import GNNExplainer

nodes_per_graph_nr = 20
load_model = False
path = 'graphs_0_2'
#dataset = load_dataset(path, type_of_feat="float")
graph = nx.generators.random_graphs.barabasi_albert_graph(nodes_per_graph_nr, 1)
# Get edges of graph -----------------------------------------------------------------------------------------------
edges = list(graph.edges())

#Select nodes for label calculation --------------------------------------------------------------------------------
edge_idx = np.random.randint(len(edges))

node_indices = [edges[edge_idx][0], edges[edge_idx][1]]
sigmas = [0, 0.1, 0.3, 0.5]
for sigma in sigmas:

    dataset, path = generate(500, nodes_per_graph_nr, sigma, graph, node_indices)

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True)

    epoch_nr = 50

    input_dim = 1
    n_classes = 2
    if load_model:
        checkpoint = torch.load(f"{path}/checkpoint/checkpoint.pth")
        model = checkpoint['model']
        opt = checkpoint['optimizer']
        scheduler = ReduceLROnPlateau(opt, 'min')
    else:
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

    no_of_runs = 10

    pg_model = copy.deepcopy(model)
    for i in range(no_of_runs):
        train_graphs = Batch.from_data_list(train_dataset)
        z = pg_model(train_graphs, train_graphs.batch,
                get_embedding=True)
        exp = PGExplainer(pg_model, 32, epochs=20, task="graph", log=True)
        exp.train_explainer(train_graphs, z, None,
                            train_graphs.batch)
        test_graphs = Batch.from_data_list(test_dataset)
        z = pg_model(test_graphs, batch=test_graphs.batch,
                get_embedding=True)
        edge_mask = exp.explain(test_graphs, z)

        em = np.reshape(edge_mask, (len(test_dataset), -1))
        #print(em.shape)
        Path(f"{path}/{sigma}/pg_results").mkdir(parents=True, exist_ok=True)
        np.savetxt(f'{path}/{sigma}/pg_results/pg_edge_masks{i}.csv', em, delimiter=',', fmt='%.3f')
        #print(em.mean(0))


    tweak_model = copy.deepcopy(model)
    for idx in range(no_of_runs):
        print(f"idx {idx}")
        exp = GNNExplainer(tweak_model, epochs=500)
        em = exp.explain_graph_TWEAK(dataset, 0.8)
        #print(em) 
        Path(f"{path}/{sigma}/tweak_resultsfull1000").mkdir(parents=True, exist_ok=True)
        gnn_edge_masks = np.reshape(em, (len(em), -1))
        np.savetxt(f'{path}/{sigma}/tweak_resultsfull1000/gnn_edge_masks{idx}.csv', gnn_edge_masks, delimiter=',', fmt='%.3f')

    '''
    gnn_model = copy.deepcopy(model)
    gnn_edge_masks = []
    for i in range(no_of_runs):
        print(i)
        gnn_edge_masks = []
        for graph in test_dataset:
            explainer = gnnexp(gnn_model, epochs=250)
            batch = []    
            for j in range(nodes_per_graph_nr):
                batch.append(0)

            nfm, em, logits_init, logits_post = explainer.explain_graph(graph, torch.tensor(batch))
            gnn_edge_masks.append(em.detach().numpy())
        Path(f"{path}/regular_results").mkdir(parents=True, exist_ok=True)
        #gnn_edge_masks = np.reshape(em, (len(em), -1))
        np.savetxt(f'{path}/regular_results/gnn_edge_masks{i}.csv', gnn_edge_masks, delimiter=',', fmt='%.3f')

    '''
