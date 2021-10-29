import igraph
import numpy as np
from itertools import combinations
import copy

def calc_edge_importance(feat_mask, edge_indices):

	print(edge_indices.shape[2])

	for xx in edge_indices.shape[2]:
        for yy in feat_mask.size():
            # do something