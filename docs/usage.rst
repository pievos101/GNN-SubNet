Usage
=====

Here we describe the general usage of GNNSubNet.

Getting Started
---------------

To use GNNSubNet, the following general procedure is followed:

1. Create a GNNSubNet object which points to a directory containing the data you wish to analyse
2. Train a graph neural network using this data
3. Run the explainer to explain the results

In the following subsections we will go through each of these steps in detail.

Step 1: Create a new GNNSubNet object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First we import the main GNNSubNet module as ``gnn``, and then
create a new GNNSubNet object, which we will call ``g``.
The datasets can be found here: 
https://github.com/pievos101/GNN-SubNet/tree/main/TCGA

.. code-block:: python

    from GNNSubNet import GNNSubNet as gnn

    # location of the files
    loc   = "/home/bastian/GitHub/GNN-SubNet/TCGA"
    # PPI network
    ppi   = f'{loc}/KIDNEY_RANDOM_PPI.txt'
    # single-omic features
    #feats = [f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
    # multi-omic features
    feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
    # outcome class
    targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'

    # Load the multi-omics data 
    g = gnn.GNNSubNet(loc, ppi, feats, targ)
    g.summary()

Once we have created our GNNSubNet object ``g`` we can train.

Step 2: Train the GNN classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # Train the GNN classifier and validate performance on a test set
    g.train()

    # Check the performance of the classifier
    g.accuracy
    g.confusion_matrix

Step 3: Run the exlainer on the trained network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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
    g.gene_names[g.modules[0]]

    g.module_importances
