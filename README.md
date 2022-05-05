![GNNSubNetLogo](https://github.com/pievos101/GNN-SubNet/blob/main/GNNSubNet_plot.png)

# GNN-SubNet: Disease Subnetwork Detection with Explainable Graph Neural Networks

## https://www.biorxiv.org/content/10.1101/2022.01.12.475995v1

Note, a python package is under development. See the python-package branch.
https://github.com/pievos101/GNN-SubNet/tree/python-package

Meanwhile, you can follow the following steps to get GNN-SubNet running.

The main file is called 'OMICS_workflow.py'.
Within that python file you find the function 'load_OMICS_dataset()'. 
It expects the PPI network as an input, the feature matrices, as well as the outcome class. The input needs to be adjusted by the user.

The PPI network consists of three columns.
The first two columns reflect the edges between the nodes (gene names), the third column is the confidence score of the specified edge. The range of this score is [0,999], where high scores mean high confidence.

The rows of the feature matrices (e.g mRNA, and DNA Methylation) reflect the patients, whereas the columns represent the features (genes). Row names as well as column names are required!

Please see the folder "datasets/TCGA" for some sample/example files.

To execute the script simply type 'python OMICS_workflow.py' within your console.

The mentioned OMICS workflow performs GNN classification, explanations, and community detection for disease subnetwork discovery. 

After exectution of 'OMICS_workflow.py', importance scores are stored in the 'edge_mask.txt' file of the data folder. 

The detected disease subnetworks can be found within the 'communities.txt' file, and the corresponding scores within the 'communities_scores.txt' file.

Note, the mentioned community file contain the network node identifier which match the order of the gene names stored in 'gene_names.txt'

  
