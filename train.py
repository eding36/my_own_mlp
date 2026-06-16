
import numpy as np
from utils import DataLoader, MLP

NUM_RESIDUES = 21
MAX_LEN = 64
NUM_EPOCHS = 10
model = MLP(input_dim = NUM_RESIDUES, learning_rate = 0.01, max_len = MAX_LEN)
def train(csv_dir, num_epochs = NUM_EPOCHS):
    
    dataloader = DataLoader(split_ratio=0.8, batch_size = 24, csv_dir = csv_dir)
    X_train, X_val, y_train, y_val = dataloader.make_train_test_split()
    for epoch in range(num_epochs):
        for X_batch, y_batch in zip(X_train,y_train):
            y_pred_batch = model.forward(X_batch) #(B,1)
            y_batch = y_batch[:, np.newaxis]
            avg_train_loss = ((y_pred_batch-y_batch)**2).mean(axis=0)
            d_loss = 2*(y_pred_batch-y_batch)
            
            print('batch train loss:', avg_train_loss)
            model.backward(d_loss)
            
        y_pred_val = model.forward(X_val)
        avg_val_loss = ((y_pred_val-y_val)**2).mean(axis=(0,1))
        
        print('avg val_loss across all validation samples:', avg_val_loss)

if __name__ == "__main__":
    train(csv_dir = "protein_sequences.csv")
    
