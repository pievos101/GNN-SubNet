# GNN-SubNet: Subnetwork Detection with Explainable Graph Neural Networks

The main file is called 'OMICS_workflow.py'.
Within that python file you find the function 'load_OMICS_dataset()'. 
It expects the PPI network as an input, the feature matrices, as well as the outcome class. The input needs to be adjusted by the user.

The PPI network consists of three columns.
The first two colums reflect the edges between gene names, the third column is the confidence score between the specified edges. The range of this score is [0,999].

The rows of the feature matrices (e.g mRNA, and DNA Methylation) reflect the patients, whereas the columns represent the features (genes). 

The mentioned OMICS workflow performs GNN classification, explanations, and community detection for disease subnetwork discovery. 

After exectution of 'OMICS_workflow.py', importance scores are stored in the 'edge_mask.txt' file of the data folder. 

The detected disease subnetworks can be found within the 'communities.txt' file, and the corresponding scores within the 'communities_scores.txt' file.

Note, the mentioned files contain the node identifier which match the gene names stored in 'gene_names.txt'

  
