# Classifying Protein Solubility 

this is a mlp classification model built from scratch using only numpy and pandas, with the goal of predicting protein solubility from amino acid sequences.

## Overview

given a CSV of protein sequences and binary solubility labels, a feedforward neural network classifies whether a protein is soluble. forward passes, backpropagation, weight and bias updates are all implemented manually.

## Files

- [train.py](train.py) — training loop: load data, featurize data, pass through mlp, compute BCE loss, call backprop, update weights & biases
- [utils.py](utils.py) — data loading functions, one-hot encoding functions, dataloader class, MLP class, linear layer class

## Model Architecture

sequences are one-hot encoded to a fixed length of MAX_RES residues × 21 possible amino acids (20 standard + a padding token),

the network consists of a stack of linear activation layers, with ReLU activation functions between each hidden layer

- hidden layers use **ReLU** activation
- output layer uses **Sigmoid** for binary classification
- weights initialized with He initialization (`sqrt(2 / dim_in)`)
- biases for each layer initialized with np.zeros(dim_out)


## Training

- **loss:** binary cross-entropy (BCE)
- **optimizer:** mini-batch gradient descent
- **learning rate:** 0.01 (tunable)
- **batch size:** 24 (tunable)
- **epochs:** 100 (tunable)
- **train/val split:** 80/20, reshuffled each epoch with the epoch number as the random seed (% tunable)

## Data Format

the training CSV (`protein_sequences.csv`) must have two columns:

| column | description |
|---|---|
| `seq` | amino acid sequence string (e.g. `MKTAYIAKQRQISFVK...`) |
| `is_soluble` | binary label: `1` = soluble, `0` = insoluble |

sequences shorter than 64 residues are zero-padded; longer sequences are truncated at MAX_RES.

## Usage

```bash
pip install numpy pandas
python train.py
```

to use a different CSV:

```python
# in train.py
train(csv_dir="your_data.csv")
```

## Dependencies

- `numpy`
- `pandas`
