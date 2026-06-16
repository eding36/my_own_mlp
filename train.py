
import numpy as np
from utils import DataLoader
from modules import MLP

NUM_RESIDUES = 21
MAX_LEN = 64
NUM_EPOCHS = 100
model = MLP(input_dim = NUM_RESIDUES, learning_rate = 0.01, max_len = MAX_LEN)
def train(csv_dir, num_epochs = NUM_EPOCHS):
    
    dataloader = DataLoader(split_ratio=0.8, batch_size = 24, csv_dir = csv_dir)
    for epoch in range(num_epochs):
        X_train, X_val, y_train, y_val = dataloader.make_train_test_split(seed=epoch)
        for X_batch, y_batch in zip(X_train,y_train):
            y_pred_batch = model.forward(X_batch) #(B,1)
            y_batch = y_batch[:, np.newaxis]
            p = np.clip(y_pred_batch, 1e-7, 1 - 1e-7)
            avg_train_loss = -(y_batch * np.log(p) + (1 - y_batch) * np.log(1 - p)).mean() #BCE loss
            d_loss = (p - y_batch) / (p * (1 - p)) / y_batch.shape[0]

            print('batch train loss:', avg_train_loss)
            model.backward(d_loss)

        y_pred_val = model.forward(X_val)
        y_val = y_val[:,np.newaxis]
        p_val = np.clip(y_pred_val, 1e-7, 1 - 1e-7)
        avg_val_loss = -(y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val)).mean()

        print('weights',model.weights[-1])
        print('biases',model.biases[-1])
        
        print(f'Val loss at Epoch{epoch}:', avg_val_loss)

if __name__ == "__main__":
    train(csv_dir = "protein_sequences.csv")
    
