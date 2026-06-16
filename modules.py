import numpy as np
from utils import softmax

class Linear():
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self):
        scale = np.sqrt(2.0/self.dim_in)
        return np.random.randn(self.dim_in, self.dim_out) * scale, np.zeros(self.dim_out)
    
class LinearStack():

    def __init__(self, input_dim, first_layer_dim):
        self.stack = []
        self.input_dim = input_dim
        self.first_layer_dim = first_layer_dim
    def forward(self):
        self.stack.append(Linear(dim_in=self.input_dim, dim_out = self.first_layer_dim))
        current_dim= self.first_layer_dim 
        while current_dim > 1:
            self.stack.append(Linear(dim_in = current_dim, dim_out = int(current_dim/2)))
            current_dim = int(current_dim/2)
        return self.stack , len(self.stack)

class AttentionLayer():
    def __init__(self, input_dim, attention_dim):
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.query_matrix = Linear(dim_in = self.input_dim, dim_out = attention_dim)
        self.key_matrix = Linear(dim_in = self.input_dim, dim_out = attention_dim)
        self.value_matrix = Linear(dim_in = self.input_dim, dim_out = attention_dim)
    
    def forward(self, x_batch):
        query = x_batch @ self.query_matrix   #(B,max_res,attention_dim)
        key = x_batch @ self.key_matrix
        value = x_batch @ self.value_matrix #(B,max_res,attention_dim)
        attention_score_matrix = softmax((query @ key.transpose(2,0,1))/np.sqrt(self.attention_dim)) #(B,max_res,max_res)
        attention_matrix = attention_score_matrix @ value #(B,max_res,attention_dim)

        return attention_matrix

    

class MLP():
    def __init__(self, input_dim, learning_rate, max_len):
        self.input_dim = input_dim
        self.stack, self.num_layers = LinearStack(input_dim = max_len * input_dim, first_layer_dim = 64).forward()
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
            weight_layer_gradient = layer_input.T @ preactivation_layer_gradient  # (dim_in, dim_out) #activation layer is used as layer_input unless input layer is reached
            bias_layer_gradient = preactivation_layer_gradient.sum(axis=0)  # (dim_out,)

            activation_gradient = preactivation_layer_gradient @ self.weights[i].T  # (B, dim_in)

            self.weights[i] = self.weights[i] - weight_layer_gradient * self.learning_rate
            self.biases[i] = self.biases[i] - bias_layer_gradient * self.learning_rate
