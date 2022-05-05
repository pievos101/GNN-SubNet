"""
    Graph Neural Network training utility methods

    :author: Afan Secic (main developer), Anna Saranti (corrections and refactoring)
    :copyright: © 2020 HCI-KDD (ex-AI) group
    :date: 2020-12-01
"""

import networkx as nx
import copy
import numpy as np
import torch

def check_if_graph_is_connected(edge_index):
    if type(edge_index) == torch.Tensor:
        edge_index = edge_index.numpy()
    s = list(copy.copy(edge_index[0]))
    t = list(copy.copy(edge_index[1]))

    s.extend(t)
    nodes = list(set(s))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = np.array(edge_index)

    edges = [(row[0].item(), row[1].item()) for row in edges.T]
    graph.add_edges_from(edges)
    
    return nx.is_connected(graph)


def _plain_bfs(G, source):
    """A fast BFS node generator"""
    G_adj = G.adj
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(G_adj[v])

def pass_data_iteratively(model, graphs, minibatch_size = 32):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)