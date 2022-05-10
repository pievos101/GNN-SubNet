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
create a new GNNSubNet object, which we will call ``g``:

.. code-block:: python

    from GNNSubNet import GNNSubNet as gnn

    g = GNNSubNet('/path/to/data', features='')

Once we have created our GNNSubNet object ``g`` we can train

Step 2: Train a graph neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Text.

Step 3: Run the exlainer on the trained network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Text.
