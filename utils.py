import numpy as np
import pandas as pd

def split_into_batches(X, batch_size):
    return [X[i:i+batch_size] for i in range(0, len(X), batch_size)]

def one_hot_encode(sequence: str, max_len = 64) -> np.ndarray:

    RESIDUES = list("ACDEFGHIKLMNPQRSTVWY0")
    RESIDUE_TO_IDX = {aa: i for i, aa in enumerate(RESIDUES)}
    
    seq = sequence.upper()
    if len(seq) < max_len:
        seq += "0" * (max_len-len(seq))
    if len(seq) > max_len:
        seq = seq[:max_len]
    out = np.zeros((max_len, len(RESIDUES)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in RESIDUE_TO_IDX:
            out[i, RESIDUE_TO_IDX[aa]] = 1.0
        # unknown residues (X, -, etc.) stay as all-zeros row
    return out

def softmax(x):
    x_stable = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x_stable)
    return e / e.sum(axis=-1, keepdims=True)

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
        split = int(self.split_ratio * len(encoded_seqs))
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, X_test = encoded_seqs[train_idx], encoded_seqs[test_idx] #(N,64)
        Y_train, Y_test = solubility_labels[train_idx], solubility_labels[test_idx]
        X_train = split_into_batches(X_train, self.batch_size) #(B,N,64)
        Y_train = split_into_batches(Y_train, self.batch_size) #(B,1)
        
        return X_train, X_test, Y_train, Y_test
    
def positional_encoding(max_len,d_model):
    pos_enc = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]  # [max_len, 1]
    i = np.arange(0, d_model, 2)  # [d_model/2]  — even indices
    div_term = 10000 ** (i / d_model)  # [d_model/2]
    
    pos_enc[:, 0::2] = np.sin(pos / div_term) # even columns: 0,2,4,...
    pos_enc[:, 1::2] = np.cos(pos / div_term) # odd columns:  1,3,5,...
    
    return pos_enc

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)