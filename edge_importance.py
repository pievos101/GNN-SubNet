import igraph
import numpy as np
from itertools import combinations
import copy

def calc_edge_importance(feat_mask, edge_indices):

	rows, cols = edge_indices

	return((feat_mask[rows]+feat_mask[cols])/2)
