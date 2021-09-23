## GNN-SubNet: Subnetwork Detection with Explainable Graph Neural Networks

DATA STRUCTURE

Each dataset needs to be in a specific folder which can be named however user likes it, but it has to end with node indices of selected nodes which compute class for a graph e.g graphs_1_2, anything_9_12 etc. 

Inside that folder there is a folder called dataset which containts all the graphs. Each graph is saved in 2 different files. One is for edges, the other is for node features. Those files have a specific naming convention so they have to be named like graph{idx}_edges.txt and graph{idx}_features.txt where idx is an index of a graph. Edge files is constructed so edges file contain graph connectivity in COO format with shape [2, num_edges] and type torch.long. Features file contains node feature matrix with shape [num_nodes, num_node_features]. 

Results from PGExplainer are stored in folder pg_results and they are organized in separate files, they are named pg_edge_masks{idx}.csv where idx is an index of a run of explainer. Same is for results from our explainer, only difference is name gnn_edge_masks{idx}.csv

All explainers can be tested in files pg_test.py, tweak_test.py, or test.py. File for constructing plots such as coverage plots, graph plots, errorbar plots etc. is in file plot_means.py only the initial path needs to be changed where dataset and edge_masks are stored. 
