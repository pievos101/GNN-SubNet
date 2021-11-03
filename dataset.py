"""
    Generating dataset for GNN

    :author: Afan Secic (main developer), Anna Saranti (corrections and refactoring)
    :copyright: © 2020 HCI-KDD (ex-AI) group
    :date: 2020-12-01
"""

import copy
import os
from pathlib import Path
import networkx as nx
from networkx.utils import arbitrary_element
from networkx.algorithms import isolate
import pandas as pd
import dgl
import dgl.data
import numpy as np
import torch
from torch._C import Graph
from torch_geometric.data.data import Data

from gnn_training_utils import _plain_bfs
from sklearn.preprocessing import minmax_scale
from features_computation import get_genes, get_genes_bernoulli, sigmoid, gen_syn_data, gen_data_community
from graph_dataset import GraphDataset
from s2vgraph import S2VGraph
from gnn_training_utils import check_if_graph_is_connected

def generate_community(graphs_nr: int, nodes_per_graph_nr: int, sigma, graph, node_indices, no_of_features):
    edges = torch.zeros(size=(2,len(graph.edges())), dtype=torch.long)
    for e, idx in zip(graph.edges(), range(len(graph.edges()))):
        edges[0][idx] = e[0]
        edges[1][idx] = e[1]
    
    genes, target_labels_all_graphs, node_indices = gen_data_community(graphs_nr, nodes_per_graph_nr, sigma, node_indices, no_of_features)
    graph = dgl.from_networkx(graph)
    graphs = []
    feats = np.zeros(shape = (graphs_nr, nodes_per_graph_nr, no_of_features))
    for graph_idx in range(graphs_nr):
        temp_graph = copy.deepcopy(graph)
        temp_graph.ndata['feat'] = torch.tensor(np.array(genes[graph_idx]))
        feats[graph_idx] = genes[graph_idx]
        target = target_labels_all_graphs[graph_idx]
        graphs.append((temp_graph, torch.tensor(target)))


    # Generating the dataset in needed form 
    data_graphs = []
    for graph_idx in range(graphs_nr):
        data_graphs.append(Data(x = torch.transpose(torch.Tensor([feats[graph_idx]]),0,1)[:,0,:],
                                edge_index = edges, 
                                y = torch.tensor(target_labels_all_graphs[graph_idx], dtype = torch.long)))

    dataset = GraphDataset(data_graphs)
    create_files(dataset, node_indices)
    path = f"graphs_{node_indices[0]}_{node_indices[1]}"
    return dataset, path

def generate(graphs_nr: int, nodes_per_graph_nr: int, sigma, graph, node_indices, no_of_features):
    """
    Generates BA graphs, assigns features to graphs, and returns GraphDataset and path
    :param graphs_nr: Number of graphs
    :param nodes_per_graph_nr: Number of nodes per graph
    :param sigma: Sigma for normal distribution
    :param graph: Graph to be used for dataset
    :param node_indices: 
    return 
    :dataset: Dataset of graphs
    :path: path where dataset is stored
    """
    #graph = nx.to_directed(graph)
    edges = torch.zeros(size=(2,len(graph.edges())), dtype=torch.long)
    for e, idx in zip(graph.edges(), range(len(graph.edges()))):
        edges[0][idx] = e[0]
        edges[1][idx] = e[1]
    
    genes, target_labels_all_graphs, node_indices = gen_syn_data(graphs_nr, nodes_per_graph_nr, sigma, node_indices, no_of_features)
    graph = dgl.from_networkx(graph)
    graphs = []
    feats = np.zeros(shape = (graphs_nr, nodes_per_graph_nr, no_of_features))
    for graph_idx in range(graphs_nr):
        temp_graph = copy.deepcopy(graph)
        temp_graph.ndata['feat'] = torch.tensor(np.array(genes[graph_idx]))
        feats[graph_idx] = genes[graph_idx]
        target = target_labels_all_graphs[graph_idx]
        graphs.append((temp_graph, torch.tensor(target)))


    # Generating the dataset in needed form 
    data_graphs = []
    for graph_idx in range(graphs_nr):
        data_graphs.append(Data(x = torch.transpose(torch.Tensor([feats[graph_idx]]),0,1)[:,0,:],
                                edge_index = edges, 
                                y = torch.tensor(target_labels_all_graphs[graph_idx], dtype = torch.long)))

    dataset = GraphDataset(data_graphs)
    create_files(dataset, node_indices)
    path = f"graphs_{node_indices[0]}_{node_indices[1]}"
    return dataset, path


def generate_simple(graphs_nr: int, nodes_per_graph_nr: int, graph: Graph, node_indices):
    """
    Generates BA graphs, assigns features to graphs, and returns GraphDataset and path
    :param graphs_nr: Number of graphs
    :param nodes_per_graph_nr: Number of nodes per graph
    :param graph: Graph to be used for dataset
    :param node_indices: 
    return 
    :dataset: Dataset of graphs
    :path: path where dataset is stored
    """

    edges = torch.zeros(size=(2,len(graph.edges())), dtype=torch.long)
    for e, idx in zip(graph.edges(), range(len(graph.edges()))):
        edges[0][idx] = e[0]
        edges[1][idx] = e[1]

    genes, target_labels_all_graphs, node_indices = get_genes_bernoulli(graphs_nr, nodes_per_graph_nr,graph)
    
    #genes, target_labels_all_graphs, node_indices = gen_syn_data(graphs_nr, nodes_per_graph_nr, 0, node_indices)
    graph = dgl.from_networkx(graph)
    graphs = []
    feats = np.zeros(shape = (graphs_nr, nodes_per_graph_nr))
    for graph_idx in range(graphs_nr):
        temp_graph = copy.deepcopy(graph)
        temp_graph.ndata['feat'] = torch.tensor(np.array(genes[graph_idx]))
        feats[graph_idx] = genes[graph_idx]
        target = target_labels_all_graphs[graph_idx]
        graphs.append((temp_graph, torch.tensor(target)))


    # Generating the dataset in needed form 
    data_graphs = []
    for graph_idx in range(graphs_nr):
        data_graphs.append(Data(x = torch.transpose(torch.Tensor([feats[graph_idx]]),0,1),
                                edge_index = edges, 
                                y = torch.tensor(target_labels_all_graphs[graph_idx], dtype = torch.long)))

    dataset = GraphDataset(data_graphs)
    create_files(dataset, node_indices)
    path = f"graphs_{node_indices[0]}_{node_indices[1]}"
    return dataset, path


def create_files(dataset: GraphDataset, indices: list):
    """
    Creates files with node features and edge index
    :param dataset: Dataset
    :param indices: Indices of nodes for computation of class
    return 
    """
    Path(f"graphs_{indices[0]}_{indices[1]}/dataset").mkdir(parents=True, exist_ok=True)
    for idx, graph in zip(range(len(dataset)),dataset):
        path_edge = f'graphs_{indices[0]}_{indices[1]}/dataset/graph{idx}_edges.txt'
        np.savetxt(path_edge, graph.edge_index.numpy(), fmt='%i')
        path_feat = f'graphs_{indices[0]}_{indices[1]}/dataset/graph{idx}_features.txt'
        np.savetxt(path_feat, graph.x.numpy(), fmt='%.3f')


def load_syn_dataset(path: str, indices=None, type_of_feat="int"):
    """
    Loads the dataset from a given path
    :param path: path to dataset
    :param indices: Indices of nodes for computation of class
    :param type_of_feat: type of features (int, float)
    return 
    :dataset: loaded dataset
    """
    if indices is None:
        last_idx = path.rindex('_')
        second_last_idx = path[:last_idx].rindex('_')
        idx1 = int(path[second_last_idx+1:last_idx])
        idx2 = int(path[last_idx+1:])
        indices = [idx1, idx2]

    file_names = os.listdir(f"{path}/dataset")
    no_of_files = len(list(file_names))
    graphs = []
    no_of_graphs = int(no_of_files/2)
    for idx in range(no_of_graphs):
        edge_path = f"{path}/dataset/graph{idx}_edges.txt"
        feat_path = f"{path}/dataset/graph{idx}_features.txt"
        edge_index = np.loadtxt(edge_path)
        features = np.loadtxt(feat_path)
        if type_of_feat=="int":
            graphs.append(Data(x=torch.tensor(features).float(),
                               edge_index=torch.tensor(edge_index, dtype=torch.long), 
                               y=torch.tensor(int(features[indices[0]]) ^ int(features[indices[1]]), dtype=torch.long)))
        else:
            if idx < int(no_of_graphs/2):
                graphs.append(Data(x=torch.tensor(features).float(),
                               edge_index=torch.tensor(edge_index, dtype=torch.long), 
                               y=torch.tensor(0), dtype=torch.long))
            else:
                graphs.append(Data(x=torch.tensor(features).float(),
                               edge_index=torch.tensor(edge_index, dtype=torch.long), 
                               y=torch.tensor(1), dtype=torch.long))        

    dataset = GraphDataset(graphs)
    return dataset


def save_results(path: str, confusion_array: list, gnn_edge_masks: list,
                 log_logits_init: list, log_logits_post: list):
    """
    Saves results of explanations to path
    :param path: path where to store results
    :param confusion_array: array of values from confusion matrix
    :param: edge_masks: values of edge masks from explainer
    return 
    """
    Path(f"{path}/results").mkdir(parents=True, exist_ok=True)

    np.savetxt(f'{path}/results/confusions.csv', confusion_array, delimiter=',', fmt="%s")
    gnn_edge_masks = np.reshape(gnn_edge_masks, (len(gnn_edge_masks), -1))
    np.savetxt(f'{path}/results/gnn_edge_masks.csv', gnn_edge_masks, delimiter=',', fmt='%.3f')
    log_logits_init = np.reshape(log_logits_init, (len(log_logits_init), -1))
    np.savetxt(f'{path}/results/log_logits_init.csv', log_logits_init, delimiter=',', fmt='%.3f')
    log_logits_post = np.reshape(log_logits_post, (len(log_logits_post), -1))
    np.savetxt(f'{path}/results/log_logits_post.csv', log_logits_post, delimiter=',', fmt='%.3f')

    
def load_KIRC_dataset(edge_path="", feat_paths=[], survival_path=""):
    """
    Loads KIRC dataset with given edge, features, and survival paths. Returns formatted dataset for further usage
    :param edge_path: String with path to file with edges
    :param feat_paths: List of strings with paths to node features
    :param survival_path: String with path to file with graph classes
    return 
    :graphs: formatted dataset
    :row_pairs: mapping between integers and proteins
    :col_pairs: mapping between integers and proteins
    """

    feats = []
    for path in feat_paths:
        feats.append(pd.read_csv(path, delimiter=' '))

    ppi_path = edge_path
    ppi = pd.read_csv(ppi_path, delimiter=" ")
    # added just for reduced number of edges
    ppi = ppi[ppi.combined_score >= 995]

    protein1 = list(set(ppi[ppi.columns.values[0]]))
    protein2 = list(set(ppi[ppi.columns.values[1]]))
    protein1.extend(protein2)
    proteins = list(set(protein1))

    nans = []
    for feat in feats:
        nans.extend(feat.columns[feat.isna().any()].tolist())
    nans = list(set(nans))

    for i in range(len(feats)):
        feats[i] = feats[i][feats[i].columns.intersection(proteins)]
        feats[i] = feats[i][feats[i].columns.difference(nans)]

    proteins = list(set(proteins) & set(feats[0].columns.values))

    old_cols = feats[0].columns.values    
    old_rows = feats[0].index.values
    new_cols = pd.factorize(old_cols)[0]
    new_rows = pd.factorize(old_rows)[0]

    col_pairs = {name: no for name,no in zip(old_cols, new_cols)}
    row_pairs = {name: no for name,no in zip(old_rows, new_rows)}

    ppi = ppi[ppi[ppi.columns.values[0]].isin(old_cols)]
    ppi = ppi[ppi[ppi.columns.values[1]].isin(old_cols)]

    ppi[ppi.columns.values[0]] = ppi[ppi.columns.values[0]].map(col_pairs)
    ppi[ppi.columns.values[1]] = ppi[ppi.columns.values[1]].map(col_pairs)    

    graphs = []
    edge_index = ppi[[ppi.columns.values[0], ppi.columns.values[1]]].to_numpy()
    edge_index = np.array(sorted(edge_index, key = lambda x: (x[0], x[1]))).T

    #first_idx = edge_path.index('/')
    #np.savetxt(f'{edge_path[:first_idx]}/edge_index.txt', edge_index, fmt='%d')

    last_idx = edge_path.rindex('/')
    
    np.savetxt(f'{edge_path[:last_idx]}/edge_index.txt', edge_index, fmt='%d')

    s = list(copy.copy(edge_index[0]))
    t = list(copy.copy(edge_index[1]))

    s.extend(t)
    nodes = list(col_pairs.values())
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = np.array(edge_index)

    edges = [(row[0].item(), row[1].item()) for row in edges.T]
    graph.add_edges_from(edges)

    nodes = []
    while len(nodes) < 1000:
        nodes = _plain_bfs(graph, np.random.randint(0,len(graph.nodes)))

        nodes = list(nodes)

    isolated_nodes = list(nx.isolates(graph))
    
    col_pairs_for_iso = {no: name for name,no in zip(old_cols, new_cols)}

    iso_nodes = [col_pairs_for_iso[x] for x in isolated_nodes]
    for feat in feats:
        feat.drop(columns = iso_nodes, inplace=True)
    
    ppi = ppi.drop(ppi[(~ppi.protein1.isin(nodes)) | (~ppi.protein2.isin(nodes))].index)
    
    drop_nodes = [col_pairs_for_iso[x] for x in nodes]
    for feat in feats:
        feat.drop(feat.columns.difference(drop_nodes), 1, inplace=True)

    new_nodes = list(range(len(nodes)))
    new_nodes_dict = {old: new for old,new in zip(nodes, new_nodes)}
    ppi[ppi.columns.values[0]] = ppi[ppi.columns.values[0]].map(new_nodes_dict)
    ppi[ppi.columns.values[1]] = ppi[ppi.columns.values[1]].map(new_nodes_dict)   
    for feat in feats:
        feat.rename(columns=new_nodes_dict, inplace=True)

    edge_index = ppi[[ppi.columns.values[0], ppi.columns.values[1]]].to_numpy()
    edge_index = np.array(sorted(edge_index, key = lambda x: (x[0], x[1]))).T

    #first_idx = edge_path.index('/')
    #np.savetxt(f'{edge_path[:first_idx]}/edge_index.txt', edge_index, fmt='%d')

    last_idx = edge_path.rindex('/')
    np.savetxt(f'{edge_path[:last_idx]}/edge_index.txt', edge_index, fmt='%d')


    temp = np.stack(feats, axis=-1)
    new_temp = []
    for item in temp:
        new_temp.append(minmax_scale(item))
    
    temp = np.array(new_temp)

    survival = pd.read_csv(survival_path, delimiter=' ')
    survival_values = survival.to_numpy()

    for idx in range(temp.shape[0]):
        graphs.append(Data(x=torch.tensor(temp[idx]).float(),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        y=torch.tensor(survival_values[0][idx], dtype=torch.long)))
        #graphs.append(Data(node_features=torch.tensor(temp[idx]).float(),
        #                edge_mat=torch.tensor(edge_index, dtype=torch.long),
        #                y=torch.tensor(survival_values[0][idx], dtype=torch.long)))
    
    return graphs, col_pairs, row_pairs

def convert_to_s2vgraph(graphs):
    s2v_graphs = []
    for graph in graphs:
        edge_index = graph.edge_index
        if type(edge_index) == torch.Tensor:
            edge_index = edge_index.numpy()

        s = list(copy.copy(edge_index[0]))
        t = list(copy.copy(edge_index[1]))

        s.extend(t)
        nodes = list(set(s))
        g = nx.Graph()
        g.add_nodes_from(nodes)
        edges = np.array(edge_index)

        edges = [(row[0].item(), row[1].item()) for row in edges.T]
        g.add_edges_from(edges)

        edge_index = torch.zeros(size=(2,len(g.edges())), dtype=torch.long)
        for e, idx in zip(g.edges(), range(len(g.edges()))):
            edge_index[0][idx] = e[0]
            edge_index[1][idx] = e[1]
        
        neighbors = []
        degrees = []
        for n in nodes:
            neighbors.append(list(g.neighbors(n)))
            degrees.append(len(neighbors[-1]))
        
        max_neighbor = max(degrees)
        s2v_graphs.append(S2VGraph(g, graph.y.item(), None, graph.x, edge_index, max_neighbor, neighbors))
    
    return s2v_graphs
