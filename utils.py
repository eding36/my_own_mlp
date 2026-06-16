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
        self.csv_dir = csv_dir
    
    def make_train_test_split(self):
        csv = pd.read_csv(self.csv_dir)
        prot_seqs = csv['seq'].tolist()
        encoded_seqs = np.array([one_hot_encode(seq) for seq in prot_seqs])
        solubility_labels = np.array(csv['is_soluble'])
        idx = np.random.permutation(len(encoded_seqs))
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
        return np.random.randn(self.dim_in, self.dim_out), np.zeros(self.dim_out)
    
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
        self.stack, self.num_layers = LinearStack(input_dim = input_dim).forward()
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
        x = x_batch
        i=0
        for layer_weights, layer_biases in zip(self.weights, self.biases):            
            x = x @ layer_weights + layer_biases
            x = 1/(1+np.exp(-x)) ##sigmoid activation fxn
            self.activations[i] = x
            i+=1

        y_pred_batch = x.mean(axis=1) #(B,1)
       
        return y_pred_batch
         
    
    def backward(self, d_loss):
        meanpool_activation_gradient = d_loss #(24,1)   
        
        last_activation_layer_gradient = np.repeat(meanpool_activation_gradient[:, None, :]/self.max_len, self.max_len, axis=1) #(24,64,1)
        activation_gradient = last_activation_layer_gradient #(24,64,1)
        
        for i in range(self.num_layers-1, 0, -1):
            #dims are for the first layer in the iteration
            activation_layer = self.activations[i] #(24,64,1)
            
            preactivation_layer_gradient = activation_gradient*((activation_layer)*(1-activation_layer)) #(24,64,1)
            weight_layer_gradient = np.einsum('bti,bto->io', self.activations[i-1], preactivation_layer_gradient) #(2,1)
            bias_layer_gradient = preactivation_layer_gradient.sum(axis=(0,1)) #(1)

            activation_gradient = preactivation_layer_gradient @ self.weights[i].T #(24,64,2)

            self.weights[i] = self.weights[i] - weight_layer_gradient*self.learning_rate
            self.biases[i] = self.biases[i] - bias_layer_gradient*self.learning_rate
            

       
            

        
        
           
