import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from numpy.core.fromnumeric import sort
import networkx as nx
from math import sqrt

graphs = 'graphs_1_3'
no_of_runs = 20

def visualize_graph(node_idx, edge_index, edge_mask, graphs, dev, exp, y=None,
                           threshold=None,**kwargs):
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.
        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.
        :rtype: :class:`matplotlib.pyplot`
        """

        assert edge_mask.size(0) == edge_index.size(1)
        
        if threshold is not None:
            print('Edge Threshold:',threshold)
            edge_mask = (edge_mask >= threshold).to(torch.float)
          
        if node_idx is  None:
            subset=[]
            for index,mask in enumerate(edge_mask):
                node_a = edge_index[0,index]
                node_b = edge_index[1,index]
                if node_a not in subset:
                    subset.append(node_a.cpu().item())
        #                     print("add: "+node_a)
                if node_b not in subset:
                    subset.append(node_b.cpu().item())
        #                     print("add: "+node_b)
        #             subset = torch.cat(subset).unique()
        edge_list=[]
        for index, edge in enumerate(edge_mask):
            if edge:
                edge_list.append((edge_index[0,index].cpu(),edge_index[1,index].cpu()))
        
        if y is None:
            y = torch.zeros(int(edge_index.max().item() + 1))
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index.cpu(), att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')

        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        #         mapping = {k: i for k, i in enumerate(subset.tolist())}
        mapping = {k: i for k, i in enumerate(subset)}
        #         print(mapping)
        #         G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 200
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
        #                     connectionstyle="arc3,rad=0.1",
                ))
        # #         if node_feature_mask is not None:
        nx.draw_networkx_nodes(G, pos, **kwargs)

        color = np.array(edge_mask.cpu())

        nx.draw_networkx_edges(G, pos,
                       width=3, alpha=0.5, edge_color=color,edge_cmap=plt.cm.Reds)
        nx.draw_networkx_labels(G, pos, **kwargs)
        plt.axis('off')
        plt.savefig(f"{graphs}/{exp}_graph_{dev[-1]}", dpi=1000)
        plt.clf()
    
def plot_gnn_explainer(graphs, dev, gnn_edge_masks, edges, edge_index):
    gnn_edge_masks = np.array(gnn_edge_masks)
    
    mins = gnn_edge_masks.min(0)
    maxes = gnn_edge_masks.max(0)
    means = gnn_edge_masks.mean(0)

    sorted_edges = [x for _, x in sorted(zip(means,edges))]
    sorted_maxes = np.array([x for _, x in sorted(zip(means, maxes))])
    sorted_mins = np.array([x for _, x in sorted(zip(means, mins))])
    sorted_means = sorted(means)

    visualize_graph(None, torch.tensor(edge_index, dtype=torch.long), torch.tensor(means), graphs, dev, "gnn")

    plt.errorbar(np.arange(len(edges)), sorted_means, fmt='ok', lw=3)
    plt.xticks(list(range(len(edges))), sorted_edges, size='small', rotation='vertical')
    plt.ylim(-0.2,1.1)
    plt.errorbar(np.arange(gnn_edge_masks.shape[1]), sorted_means, [sorted_means - sorted_mins, sorted_maxes - sorted_means], fmt='.k', ecolor='gray', lw=1)
    plt.savefig(f"{graphs}/gnn_explainer_errorbar_{dev[-1]}", dpi=1000)
    plt.clf()

def load_gnn_explainer(graphs, dev):
    last_idx = graphs.rindex('_')
    second_last_idx = graphs[:last_idx].rindex('_')
    idx1 = int(graphs[second_last_idx+1:last_idx])
    idx2 = int(graphs[last_idx+1:])
    edge = f"{idx1}-{idx2}"
    gnn_edge_masks = []
    path = f"{graphs}/dataset/graph0_edges.txt"
    edges = np.loadtxt(path).T
    edge_index = np.loadtxt(path)
    edges = list(map(lambda x: f'{int(x[0])}-{int(x[1])}', edges))

    count5, count3, count = 0, 0, 0
    for i in range(no_of_runs):
        path = f"{graphs}/{dev}/gnn_explainer/gnn_edge_masks{i}.csv"
        temp = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)
        temp = temp.T
        gnn_edge_masks.append(temp)
        sorted_edges = [x for _, x in sorted(zip(temp,edges), reverse=True)]
        index = sorted_edges.index(edge)
        if index == 0:
            count5 += 1
            count3 += 1
            count += 1
        elif index < 3:
            count3 += 1
            count5 += 1
        elif index < 5:
            count5 += 1
    
    plot_gnn_explainer(graphs, dev, gnn_edge_masks, edges, edge_index)
    return [count/no_of_runs, count3/no_of_runs, count5/no_of_runs]

def plot_gnn_modified(graphs, dev, gnn_edge_masks, edges, edge_index):
    gnn_edge_masks = np.array(gnn_edge_masks)
    
    mins = gnn_edge_masks.min(0)
    maxes = gnn_edge_masks.max(0)
    means = gnn_edge_masks.mean(0)

    sorted_edges = [x for _, x in sorted(zip(means,edges))]
    sorted_maxes = np.array([x for _, x in sorted(zip(means, maxes))])
    sorted_mins = np.array([x for _, x in sorted(zip(means, mins))])
    sorted_means = sorted(means)

    visualize_graph(None, torch.tensor(edge_index, dtype=torch.long), torch.tensor(means), graphs, dev, "gnn_modified")

    plt.errorbar(np.arange(len(edges)), sorted_means, fmt='ok', lw=3)
    plt.xticks(list(range(len(edges))), sorted_edges, size='small', rotation='vertical')
    plt.ylim(-0.2,1.1)
    plt.errorbar(np.arange(gnn_edge_masks.shape[1]), sorted_means, [sorted_means - sorted_mins, sorted_maxes - sorted_means], fmt='.k', ecolor='gray', lw=1)
    plt.savefig(f"{graphs}/gnn_modified_errorbar_{dev[-1]}", dpi=1000)
    plt.clf()

def load_modified_gnn(graphs, dev):
    last_idx = graphs.rindex('_')
    second_last_idx = graphs[:last_idx].rindex('_')
    idx1 = int(graphs[second_last_idx+1:last_idx])
    idx2 = int(graphs[last_idx+1:])
    edge = f"{idx1}-{idx2}"
    gnn_edge_masks = []
    path = f"{graphs}/dataset/graph0_edges.txt"
    edges = np.loadtxt(path).T
    edge_index = np.loadtxt(path)
    edges = list(map(lambda x: f'{int(x[0])}-{int(x[1])}', edges))

    count5, count3, count = 0, 0, 0
    for i in range(no_of_runs):
        path = f"{graphs}/{dev}/modified_gnn/gnn_edge_masks{i}.csv"
        temp = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)
        temp = temp.T
        gnn_edge_masks.append(temp)
        sorted_edges = [x for _, x in sorted(zip(temp,edges), reverse=True)]
        index = sorted_edges.index(edge)
        if index == 0:
            count5 += 1
            count3 += 1
            count += 1
        elif index < 3:
            count3 += 1
            count5 += 1
        elif index < 5:
            count5 += 1
    
    plot_gnn_modified(graphs, dev, gnn_edge_masks, edges, edge_index)
    return [count/no_of_runs, count3/no_of_runs, count5/no_of_runs]

def plot_pg(graphs, dev, pg_edge_masks, edges, edge_index):
    pg_edge_masks = np.array(pg_edge_masks)
    mins = pg_edge_masks.min(0)
    maxes = pg_edge_masks.max(0)
    means = pg_edge_masks.mean(0)

    sorted_edges = [x for _, x in sorted(zip(means,edges))] 
    sorted_maxes = np.array([x for _, x in sorted(zip(means, maxes))])
    sorted_mins = np.array([x for _, x in sorted(zip(means, mins))])
    sorted_means = sorted(means)

    visualize_graph(None, torch.tensor(edge_index, dtype=torch.long), torch.tensor(means), graphs, dev, "pg")

    plt.errorbar(np.arange(len(edges)), sorted_means, fmt='ok', lw=3)
    plt.xticks(list(range(len(edges))), sorted_edges, size='small', rotation='vertical')
    plt.ylim(-0.2,1.1)
    plt.errorbar(np.arange(pg_edge_masks.shape[1]), sorted_means, [sorted_means - sorted_mins, sorted_maxes - sorted_means], fmt='.k', ecolor='gray', lw=1)
    plt.savefig(f"{graphs}/pg_errorbar_{dev[-1]}", dpi=1000)
    plt.clf()

def load_pg(graphs, dev):
    last_idx = graphs.rindex('_')
    second_last_idx = graphs[:last_idx].rindex('_')
    idx1 = int(graphs[second_last_idx+1:last_idx])
    idx2 = int(graphs[last_idx+1:])
    edge = f"{idx1}-{idx2}"
    pg_edge_masks = []

    path = f"{graphs}/dataset/graph0_edges.txt"
    edge_index = np.loadtxt(path)
    edges = np.loadtxt(path).T
    
    edges = list(map(lambda x: f'{int(x[0])}-{int(x[1])}', edges))

    count5, count3, count = 0, 0, 0
    for i in range(no_of_runs):
        path = f"{graphs}/{dev}/pg_results/pg_edge_masks{i}.csv"
        temp = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)
        temp = np.array(temp.T)
        means = temp.mean(1)
        pg_edge_masks.append(means)
        sorted_edges = [x for _, x in sorted(zip(means,edges), reverse=True)]
        index = sorted_edges.index(edge)
        if index == 0:
            count5 += 1
            count3 += 1
            count += 1
        elif index < 3:
            count3 += 1
            count5 += 1
        elif index < 5:
            count5 += 1
    
    plot_pg(graphs, dev, pg_edge_masks, edges, edge_index)
    return [count/no_of_runs, count3/no_of_runs, count5/no_of_runs]

def plot_multiple(graphs):
    p = Path(graphs)
    devs = sorted([x.name for x in p.iterdir() if x.is_dir()])
    if 'dataset' in devs:
        devs.remove('dataset')
    for dev in devs:
        pg = load_pg(graphs, dev)
        mod = load_modified_gnn(graphs, dev)
        gnn = load_gnn_explainer(graphs, dev)
        x = np.array([1,3,5])
        w = 0.05
        ax = plt.subplot(111)
        plt.ylabel("Occurence")
        plt.xlabel("Top ")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        ax.set_xticks(x)
        ax.set_title(f"Standard deviation {dev}")
        ax.bar(x-2*w, gnn, width=w, color='r', align='center', label='GNNExplainer')
        ax.bar(x-w, pg, width=w, color='b', align='center', label='PGExplainer')
        ax.bar(x, mod, width=w, color='g', align='center', label='Our GNNExplainer')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(f"{graphs}/coverage_{dev[-1]}", dpi=1000)
        plt.clf()

plot_multiple(graphs)