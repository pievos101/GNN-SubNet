import igraph
import numpy as np
from itertools import combinations
import copy

def find_communities(edge_index_path, edge_masks_path=None, detection_alg='louvain'):
    """
    Creates communities of nodes in a graph based on edge masks and algorithm
    :param edge_index_path: String which contains path to edge_index file
    :param edge_masks_path: String which contains path to edge_mask file
    :param detection_alg: String which decides on algorithm to be used
    return
    Average edge masks per community and communities
    """

    assert detection_alg in ['louvain', 'opt_modularity']

    edge_index = np.loadtxt(edge_index_path, dtype=int)
    if edge_masks_path is not None:
        edge_masks = abs(np.loadtxt(edge_masks_path, dtype=float))
    else:
        edge_masks = None
    s = list(copy.copy(edge_index[0]))
    t = list(copy.copy(edge_index[1]))

    s.extend(t)
    nodes = list(set(s))

    edges = np.array(edge_index)

    edges = [(row[0], row[1]) for row in edges.T]

    g = igraph.Graph()
    # use max(nodes)+1 for modified datasets for shorter runtime
    # all other cases use len(nodes)
    g.add_vertices(max(nodes)+1)
    #g.add_vertices(len(nodes))
    g.add_edges(edges)
    g.es['weight'] = edge_masks

    if detection_alg == 'louvain':
        partition = g.community_multilevel(weights=edge_masks)
    elif detection_alg == 'opt_modularity':
        partition = g.community_optimal_modularity(weights=edge_masks)
    combs = []
    for i in range(len(partition)):
        cs = []
        for c in combinations(partition[i], r=2):
            cs.append(c)    
        combs.append(sorted(list(set(edges) & set(cs))))

    avg_edge_masks = []
    for com in combs:
        if(len(com) == 0):
            continue
        avg_mask = 0
        if edge_masks is not None:
            count = 0
            for edge in com:
                avg_mask += g.es[g.get_eid(edge[0], edge[1])]['weight']
                count = count + 1
            avg_edge_masks.append(avg_mask/count) 
        
    return avg_edge_masks, partition

