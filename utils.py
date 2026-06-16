import numpy as np
import pandas as pd
    

def split_into_batches(X, batch_size):
    return [X[i:i+batch_size] for i in range(0, len(X), batch_size)]

def one_hot_encode(sequence: str, max_len = 64) -> np.ndarray:

    RESIDUES = list("ACDEFGHIKLMNPQRSTVWY0")
    RESIDUE_TO_IDX = {aa: i for i, aa in enumerate(RESIDUES)}
    
    seq = sequence.upper()
    if len(seq) < max_len:
        seq += "0"*(max_len-len(seq))
    if len(seq) > max_len:
        seq = seq[:max_len]
    out = np.zeros((max_len, len(RESIDUES)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in RESIDUE_TO_IDX:
            out[i, RESIDUE_TO_IDX[aa]] = 1.0
        # unknown residues (X, -, etc.) stay as all-zeros row
    return out

class DataLoader():
    def __init__(self, split_ratio, batch_size, csv_dir):
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        csv = pd.read_csv(csv_dir)
        prot_seqs = csv['seq'].tolist()
        self.encoded_seqs = np.array([one_hot_encode(seq) for seq in prot_seqs])
        self.labels = np.array(csv['is_soluble'])

    def make_train_test_split(self, seed):
        encoded_seqs = self.encoded_seqs
        solubility_labels = self.labels
        idx = np.random.default_rng(seed=seed).permutation(len(encoded_seqs))
        split = int(self.split_ratio*len(encoded_seqs))
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, X_test = encoded_seqs[train_idx], encoded_seqs[test_idx] #(N,64)
        Y_train, Y_test = solubility_labels[train_idx], solubility_labels[test_idx]
        X_train = split_into_batches(X_train, self.batch_size) #(B,N,64)
        Y_train = split_into_batches(Y_train, self.batch_size) #(B,1)
        
        return X_train, X_test, Y_train, Y_test
    
class Linear():
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self):
        scale = np.sqrt(2.0/self.dim_in)
        return np.random.randn(self.dim_in, self.dim_out) * scale, np.zeros(self.dim_out)
    
class LinearStack():

    def __init__(self, input_dim):
        self.stack = []
        self.input_dim = input_dim
    def forward(self):
        self.stack.append(Linear(dim_in=self.input_dim, dim_out = 64))
        current_dim=64 
        while current_dim > 1:
            self.stack.append(Linear(dim_in = current_dim, dim_out = int(current_dim/2)))
            current_dim = int(current_dim/2)
        return self.stack , len(self.stack)

class MLP():
    def __init__(self, input_dim, learning_rate, max_len):
        self.input_dim = input_dim
        self.stack, self.num_layers = LinearStack(input_dim = max_len * input_dim).forward()
        self.weights = []
        self.biases = []
        self.activations = [None]*self.num_layers
        self.learning_rate = learning_rate
        self.max_len = max_len
        

        for layer in self.stack:
            layer_weights, layer_biases = layer.forward()
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
        

    def forward(self, x_batch):
        self.input = x_batch.reshape(x_batch.shape[0], -1)  # (B, max_len*num_residues)
        x = self.input
        i=0
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            x = x @ layer_weights + layer_biases
            is_last = (i == self.num_layers - 1)
            x = 1/(1+np.exp(-x)) if is_last else np.maximum(0, x)
            self.activations[i] = x
            i+=1

        return x  # (B, 1)
         
    
    def backward(self, d_loss):
        activation_gradient = d_loss  # (B, 1)

        for i in range(self.num_layers-1, -1, -1):
            activation_layer = self.activations[i]  # (B, dim_out)

            is_last = (i == self.num_layers - 1)
            if is_last:
                act_grad = activation_layer * (1 - activation_layer) #derivative of sigmoid activation ONLY at last layer 
            else:
                act_grad = (activation_layer > 0).astype(activation_layer.dtype) #derivative of ReLU activation
            preactivation_layer_gradient = activation_gradient * act_grad  # (B, dim_out)

            layer_input = self.activations[i-1] if i > 0 else self.input  # (B, dim_in)
            weight_layer_gradient = layer_input.T @ preactivation_layer_gradient  # (dim_in, dim_out)
            bias_layer_gradient = preactivation_layer_gradient.sum(axis=0)  # (dim_out,)

            activation_gradient = preactivation_layer_gradient @ self.weights[i].T  # (B, dim_in)

            self.weights[i] = self.weights[i] - weight_layer_gradient * self.learning_rate
            self.biases[i] = self.biases[i] - bias_layer_gradient * self.learning_rate
