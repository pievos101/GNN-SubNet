from collections import Counter

import dgl
import numpy as np


def sigmoid(x: int):
    """"
    Compute sigmoid function by definition
    """
    return 1/(1+np.exp(x))

def get_genes(graphs_nr: int, nodes_per_graph_nr: int, graph: dgl.DGLGraph):
    """
    Compute the genes/features for all graphs

    :param graphs_nr: Number of graphs
    :param nodes_per_graph_nr: Number of nodes per graph
    :param graph: Graph for computation
    :return:
    """

    # Generate random integer arrays for the features of all nodes -----------------------------------------------------
    randint_gen_upper_non_incl_limit = 2   # Parameter is one above the highest such integer ---------------------------
    genes = np.random.randint(randint_gen_upper_non_incl_limit, size=(graphs_nr, nodes_per_graph_nr))
    '''
    genes = np.ones((graphs_nr, nodes_per_graph_nr), dtype=int)
    feats = np.random.randint(randint_gen_upper_non_incl_limit, size=(graphs_nr,2))
    '''
    # Get edges of graph -----------------------------------------------------------------------------------------------
    edges = list(graph.edges())
    
    #Select nodes for label calculation --------------------------------------------------------------------------------
    edge_idx = np.random.randint(len(edges))
    node_indices = [edges[edge_idx][0], edges[edge_idx][1]]

    '''
    genes[:,node_indices[0]] = feats[:,0]
    genes[:,node_indices[1]] = feats[:,1]
    '''
    #Print node indices so we know which 2 nodes we want to look in explanation
    print(f"Nodes taken into computatio of the class {str(node_indices)} with edge index {edge_idx}")
    target_labels_all_graphs = []    

    # Compute the target for each patient ------------------------------------------------------------------------------
    for gene in genes:
        # Set the label to XOR of node feature values ------------------------------------------------------------------
        target_labels_all_graphs.append(gene[node_indices[0]] ^ gene[node_indices[1]])

    # Target labels ----------------------------------------------------------------------------------------------------
    print(f"Number of unique classes: {Counter(target_labels_all_graphs).keys()}")
    print(f"Number of labels for each class: {Counter(target_labels_all_graphs).values()}")

    # "target_labels_all_graphs" should just have length (nodes_per_graph_nr,) and not (nodes_per_graph_nr, 5) ---------

    return genes, target_labels_all_graphs, node_indices


def get_genes_bernoulli(graphs_nr: int, nodes_per_graph_nr: int, graph: dgl.DGLGraph):
    """
    Compute the genes/features for all graphs

    :param graphs_nr: Number of graphs
    :param nodes_per_graph_nr: Number of nodes per graph
    :param graph: Graph for computation
    :return:
    """
    # Generate random number arrays from normal distribution for the features of all nodes -----------------------------------------------------
    mi = 0
    sigma = 1
    genes = np.random.normal(mi, sigma, size=(graphs_nr, nodes_per_graph_nr))
    '''
    genes = np.ones((graphs_nr, nodes_per_graph_nr), dtype=int)
    feats = np.random.normal(mi, sigma, size=(graphs_nr, 2))
    '''
    # Get edges of graph -----------------------------------------------------------------------------------------------
    edges = list(graph.edges())

    #Select nodes for label calculation --------------------------------------------------------------------------------
    edge_idx = np.random.randint(len(edges))
    node_indices = [edges[edge_idx][0], edges[edge_idx][1]]

    '''
    genes[:,node_indices[0]] = feats[:,0]
    genes[:,node_indices[1]] = feats[:,1]
    '''
    #Print node indices so we know which 2 nodes we want to look in explanation
    print("Nodes taken into computatio of the class " + str(node_indices))
    
    target_labels_all_graphs = []

    # Compute the target for each patient ------------------------------------------------------------------------------
    for gene in genes:
        # Set the label to a sample from Bernoulli distribution ------
        z = 1 + 2*gene[node_indices[0]] + 3*gene[node_indices[1]]
        prob = sigmoid(z)
        target = np.random.binomial(1,prob)
        target_labels_all_graphs.append(target)

    # Target labels ----------------------------------------------------------------------------------------------------
    print(f"Number of unique classes: {Counter(target_labels_all_graphs).keys()}")
    print(f"Number of labels for each class: {Counter(target_labels_all_graphs).values()}")

    # "target_labels_all_graphs" should just have length (nodes_per_graph_nr,) and not (nodes_per_graph_nr, 5) ---------
    return genes, target_labels_all_graphs, node_indices


def gen_syn_data(graphs_nr: int, nodes_per_graph_nr: int, sigma, node_indices, no_of_features: int):
    """
    Compute the genes/features for all graphs

    :param graphs_nr: Number of graphs
    :param nodes_per_graph_nr: Number of nodes per graph
    :param graph: Graph for computation
    :param sigma: Sigma for normal distribution
    :param node_indices:
    :param no_of_features: Number of features per node in graph
    :return:
    """

    mi = 0
    sigma = sigma
    genes = np.random.normal(mi, sigma, size=(graphs_nr, nodes_per_graph_nr, no_of_features))

    mi = 1
    sigma = sigma #0.01
    feats = np.random.normal(mi, sigma, size=(int(graphs_nr/2), 2, no_of_features))
    '''
    # Get edges of graph -----------------------------------------------------------------------------------------------
    edges = list(graph.edges())

    #Select nodes for label calculation --------------------------------------------------------------------------------
    edge_idx = np.random.randint(len(edges))

    node_indices = [edges[edge_idx][0], edges[edge_idx][1]]
    '''
    
    genes[:int(graphs_nr/2),node_indices[0]] = feats[:,0]
    genes[:int(graphs_nr/2),node_indices[1]] = feats[:,1]
    
    #Print node indices so we know which 2 nodes we want to look in explanation
    print("Nodes taken into computatio of the class " + str(node_indices))
    
    target_labels_all_graphs = []

    # Compute the target for each patient ------------------------------------------------------------------------------
    for gene in range(int(len(genes)/2)):
        # Set the label to a sample from Bernoulli distribution ------
        target_labels_all_graphs.append(0)
    
    mi = -1
    sigma = sigma #0.01
    feats = np.random.normal(mi, sigma, size=(int(graphs_nr/2), 2, no_of_features))

    
    genes[int(graphs_nr/2):,node_indices[0]] = feats[:,0]
    genes[int(graphs_nr/2):,node_indices[1]] = feats[:,1]
    
    for gene in range(int(len(genes)/2)):
        # Set the label to a sample from Bernoulli distribution ------
        target_labels_all_graphs.append(1)

    # Target labels ----------------------------------------------------------------------------------------------------
    print(f"Number of unique classes: {Counter(target_labels_all_graphs).keys()}")
    print(f"Number of labels for each class: {Counter(target_labels_all_graphs).values()}")

    # "target_labels_all_graphs" should just have length (nodes_per_graph_nr,) and not (nodes_per_graph_nr, 5) ---------
    
    return genes, target_labels_all_graphs, node_indices


def gen_data_community(graphs_nr: int, nodes_per_graph_nr: int, sigma, node_indices, no_of_features: int):
    mi = 0
    sigma = sigma
    genes = np.random.normal(mi, sigma, size=(graphs_nr, nodes_per_graph_nr, no_of_features))

    mi = 1
    sigma = 0.01
    feats = np.random.normal(mi, sigma, size=(int(graphs_nr/2), len(node_indices), no_of_features))
    '''
    # Get edges of graph -----------------------------------------------------------------------------------------------
    edges = list(graph.edges())

    #Select nodes for label calculation --------------------------------------------------------------------------------
    edge_idx = np.random.randint(len(edges))

    node_indices = [edges[edge_idx][0], edges[edge_idx][1]]
    '''
    
    for idx, i in zip(node_indices, range(len(node_indices))):
        genes[:int(graphs_nr/2),idx] = feats[:,i]
    
    #Print node indices so we know which 2 nodes we want to look in explanation
    print("Nodes taken into computatio of the class " + str(node_indices))
    
    target_labels_all_graphs = []

    # Compute the target for each patient ------------------------------------------------------------------------------
    for gene in range(int(len(genes)/2)):
        
        target_labels_all_graphs.append(0)
    
    mi = -1
    sigma = 0.01
    feats = np.random.normal(mi, sigma, size=(int(graphs_nr/2), len(node_indices), no_of_features))

    
    for idx, i in zip(node_indices, range(len(node_indices))):
        genes[:int(graphs_nr/2),idx] = feats[:,i]
    
    for gene in range(int(len(genes)/2)):
        
        target_labels_all_graphs.append(1)

    # Target labels ----------------------------------------------------------------------------------------------------
    print(f"Number of unique classes: {Counter(target_labels_all_graphs).keys()}")
    print(f"Number of labels for each class: {Counter(target_labels_all_graphs).values()}")

    # "target_labels_all_graphs" should just have length (nodes_per_graph_nr,) and not (nodes_per_graph_nr, 5) ---------
    
    return genes, target_labels_all_graphs, node_indices