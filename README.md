![GNNSubNetLogo](https://github.com/pievos101/GNN-SubNet/blob/main/GNNSubNet_plot.png)

# GNN-SubNet: Disease Subnetwork Detection with Explainable Graph Neural Networks

## Paper 

The paper for this project is available here: <https://www.biorxiv.org/content/10.1101/2022.01.12.475995v1> 

## Installation

To install GNNSubNet run:

```python
pip install GNNSubNet
```
## Usage

### Synthetic Barabasi Networks

The datasets can be found here: 
https://github.com/pievos101/GNN-SubNet/tree/main/GNNSubNet/datasets/synthetic

```python
from GNNSubNet import GNNSubNet as gnn

# Synthetic data set  ------------------------- #
loc   = "/home/bastian/GitHub/GNN-SubNet/GNNSubNet/datasets/synthetic"
ppi   = f'{loc}/NETWORK_synthetic.txt'
feats = [f'{loc}/FEATURES_synthetic.txt']
targ  = f'{loc}/TARGET_synthetic.txt'

# Read in the synthetic data
g = gnn.GNNSubNet(loc, ppi, feats, targ, normalize=False)

# Get some general information about the data dimension
g.summary()

# Train the GNN classifier and validate performance on a test set
g.train()

# Check the performance of the classifier
g.accuracy
g.confusion_matrix

# Run the Explainer with 4 iterations (10 is recommended)
g.explain(3)

# Edge and Node (Gene) Importances 
g.gene_names
g.edges

# Importances
g.edge_mask
g.node_mask

# Detected modules and their importances
g.modules
g.modules[0]

g.module_importances

```

### TCGA multi-omics kidney cancer
The datasets can be found here:
https://github.com/pievos101/GNN-SubNet/tree/main/TCGA

```python
from GNNSubNet import GNNSubNet as gnn

loc   = "/home/bastian/GitHub/GNN-SubNet/TCGA"
ppi   = f'{loc}/KIDNEY_RANDOM_PPI.txt'
#feats = [f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'

# Load the multi-omics data 
g = gnn.GNNSubNet(loc, ppi, feats, targ)

# Train the GNN classifier and validate performance on a test set
g.train()

# Check the performance of the classifier
g.accuracy
g.confusion_matrix

# Run the Explainer with 4 iterations (10 is recommended)
g.explain(3)

# Edge and Node (Gene) Importances 
g.gene_names
g.edges

# Importances
g.edge_mask
g.node_mask

# Detected modules and their importances
g.modules
g.modules[0]

g.module_importances

```

## Further informations

The GNNSubNet initialization function  expects the PPI network as an input, the feature matrices, as well as the outcome class. The input needs to be adjusted by the user.

The PPI network consists of three columns.
The first two columns reflect the edges between the nodes (gene names), the third column is the confidence score of the specified edge. The range of this score is [0,999], where high scores mean high confidence.

The rows of the feature matrices (e.g mRNA, and DNA Methylation) reflect the patients, whereas the columns represent the features (genes). Row names as well as column names are required!

Please see the folder "datasets/TCGA" for some sample/example files.

The mentioned workflow performs GNN classification, explanations, and community detection for disease subnetwork discovery. 

After exectution, importance scores are stored in the 'edge_mask.txt' file of the data folder. 

The detected disease subnetworks can be found within the 'communities.txt' file, and the corresponding scores within the 'communities_scores.txt' file.

Note, the mentioned community file contain the network node identifier which match the order of the gene names stored in 'gene_names.txt'

  
