"""
    Graph Neural Network training utility methods

    :author: Afan Secic (main developer), Anna Saranti (corrections and refactoring)
    :copyright: © 2020 HCI-KDD (ex-AI) group
    :date: 2020-12-01
"""

import dgl
import torch


def collate(samples):
    """
    Process the list of samples to form a batch

    :param samples:
    :return:
    """

    # labels = []
    # for graph in samples:
    #    count = 0
    #    for item in graph.ndata['label']:
    #        count = count + item
    #    if count > 4:
    #        labels.append(1)
    #    else:
    #        labels.append(0)

    # graphs = samples
    # batched_graph = dgl.batch(graphs)
    # batched_labels = torch.tensor(labels)

    graphs, labels = map(list, zip(*samples))

    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)

    return batched_graph, batched_labels

