class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None, edge_mat=None, max_neighbor=0, neighbors=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = neighbors
        self.node_features = node_features
        self.edge_mat = edge_mat

        self.max_neighbor = max_neighbor