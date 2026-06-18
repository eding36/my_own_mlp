import numpy as np
from utils import softmax, positional_encoding, sigmoid, relu

#TODO: implement FFN class

class Linear():
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self):
        scale = np.sqrt(2.0/self.dim_in)
        return np.random.randn(self.dim_in, self.dim_out) * scale, np.zeros(self.dim_out) #weights, biases

    
class LayerNorm():
    def __init__(self, normalized_shape: tuple, eps: float, affine: bool, bias: bool):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine = affine
        self.bias = bias
        
        self.gamma = np.ones(self.normalized_shape[0])
        self.beta = np.zeros(self.normalized_shape[0])

    def forward(self, batch_input):
        
        mean_indices = (tuple(-(x+1) for x in range(len(self.normalized_shape))))
        mean_indices = mean_indices[::-1]
 
        mean = batch_input.mean(axis=mean_indices)
        mean = mean[:, :, np.newaxis]
        var = batch_input.var(axis=mean_indices)
        var = var[:, :, np.newaxis]
        
        x_norm = (batch_input-mean)/(np.sqrt(var+self.eps))
       
        out = x_norm * self.gamma + self.beta
        self.out = out
        self.x_norm = x_norm
        self.var = var
        self.mean = mean

        return out
    
    def backward(self, input_gradient, learning_rate):
        grad_y = input_gradient
        if self.affine:
            summation_dimension_indices = tuple(x for x in range(len(self.x_norm.shape)-len(self.normalized_shape)))
            grad_gamma = (grad_y*self.x_norm).sum(axis=summation_dimension_indices)
            grad_beta = grad_y.sum(axis=summation_dimension_indices)
            self.gamma = self.gamma - grad_gamma*learning_rate
            self.beta = self.beta - grad_beta*learning_rate
        
        grad_xhat = grad_y * self.gamma
        std_inv = 1.0 / np.sqrt(self.var + self.eps)              # [B,N,1]
        grad_x = std_inv / self.normalized_shape[0] * (
            self.normalized_shape[0] * grad_xhat
            - grad_xhat.sum(axis=-1, keepdims=True)
            - self.x_norm * (grad_xhat * self.x_norm).sum(axis=-1, keepdims=True)
        )  
        return grad_x


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
        self.query_matrix = Linear(dim_in = self.input_dim, dim_out = attention_dim).forward()[0]
        self.key_matrix = Linear(dim_in = self.input_dim, dim_out = attention_dim).forward()[0]
        self.value_matrix = Linear(dim_in = self.input_dim, dim_out = attention_dim).forward()[0]
    
    def forward(self, batch_input):
        
        query = batch_input @ self.query_matrix   #(B,max_res,attention_dim)
        key = batch_input @ self.key_matrix
        value = batch_input @ self.value_matrix #(B,max_res,attention_dim)
        
        attention_score = (query @ key.transpose(0,2,1))/np.sqrt(self.attention_dim) #(B,max_res,max_res)
        attention = softmax(attention_score)
        output = attention @ value #(B,max_res,attention_dim)

        self.output = output
        self.attention_score = attention_score
        self.attention = attention
        self.batch_input = batch_input
        self.value = value
        self.key = key
        self.query = query

        return output

    def backward(self, input_gradient, learning_rate):
        output_gradient = input_gradient #(B,max_res,attention_dim)
        value_gradient = self.attention.transpose(0,2,1) @ output_gradient #(B, max_res, attention_dim) 
        attention_gradient = output_gradient @ self.value.transpose(0,2,1) #(post softmax) #(B,max_res, max_res)
        attention_score_gradient = self.attention * (attention_gradient-(attention_gradient*self.attention)).sum(axis=-1, keepdims=True) #(pre softmax) #(B, max_res, max_res)
        attention_score_gradient = attention_score_gradient/np.sqrt(self.attention_dim)
        query_gradient = attention_score_gradient @ self.key #(B,max_res, attention_dim)
        key_gradient = attention_score_gradient.transpose(0,2,1) @ self.query #(B, max_res, attention_dim)

        query_matrix_gradient = np.einsum('bti,bto->io', self.batch_input, query_gradient)  #(input_dim, attention_dim)
        key_matrix_gradient = np.einsum('bti,bto->io', self.batch_input, key_gradient)  #(input_dim, attention_dim)
        value_matrix_gradient = np.einsum('bti,bto->io', self.batch_input, value_gradient)  #(input_dim, attention_dim)
        batch_input_gradient = query_gradient @ self.query_matrix.T + key_gradient @ self.key_matrix.T + value_gradient @ self.value_matrix.T

        self.query_matrix = self.query_matrix - query_matrix_gradient*learning_rate #update Q_w, K_w, V_w weights
        self.key_matrix = self.key_matrix - key_matrix_gradient*learning_rate
        self.value_matrix = self.value_matrix - value_matrix_gradient*learning_rate

        return batch_input_gradient
    

class MLP():
    def __init__(self, input_dim, max_len):
        self.input_dim = input_dim
        self.stack, self.num_layers = LinearStack(input_dim = max_len * input_dim, first_layer_dim = 64).forward()
        self.weights = []
        self.biases = []
        self.activations = [None]*self.num_layers
        self.max_len = max_len
        

        for layer in self.stack:
            layer_weights, layer_biases = layer.forward()
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
        

    def forward(self, batch_input):
        self.input = batch_input.reshape(batch_input.shape[0], -1)  # (B, max_len*num_residues)
        x = self.input
        i=0
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            x = x @ layer_weights + layer_biases
            is_last = (i == self.num_layers - 1)
            x = sigmoid(x) if is_last else relu(x)
            self.activations[i] = x
            i+=1

        return x  # (B, 1)
    
    
    def backward(self, input_gradient, learning_rate):
        activation_gradient = input_gradient

        for i in range(self.num_layers-1, -1, -1):
            activation_layer = self.activations[i]  # (B, dim_out)

            is_last = (i == self.num_layers - 1)
            if is_last:
                d_act = activation_layer * (1 - activation_layer) #derivative of sigmoid activation ONLY at last layer 
            else:
                d_act = (activation_layer > 0).astype(activation_layer.dtype) #derivative of ReLU activation
            preactivation_layer_gradient = activation_gradient * d_act  # (B, dim_out)

            layer_input = self.activations[i-1] if i > 0 else self.input  # (B, dim_in)
            weight_layer_gradient = layer_input.T @ preactivation_layer_gradient  # (dim_in, dim_out) #activation layer is used as layer_input unless input layer is reached
            bias_layer_gradient = preactivation_layer_gradient.sum(axis=0)  # (dim_out,)

            activation_gradient = preactivation_layer_gradient @ self.weights[i].T  # (B, dim_in)

            self.weights[i] = self.weights[i] - weight_layer_gradient * learning_rate
            self.biases[i] = self.biases[i] - bias_layer_gradient * learning_rate

        return activation_gradient

class Transformer():
    def __init__(self, max_len, input_dim, d_model):
        self.input_dim = input_dim
        self.max_len = max_len
        self.d_model = d_model

        self.input_embedding_matrix = Linear(dim_in = self.input_dim, dim_out = d_model).forward()[0] #only fetch the weight matrix, not the bias matrix
        self.pos_enc = positional_encoding(max_len, d_model)
        self.layer_norm1 = LayerNorm((d_model,), eps= 1e-5, affine = True, bias = True)
        self.layer_norm2 = LayerNorm((d_model,), eps= 1e-5, affine = True, bias = True)
        self.attention_network = AttentionLayer(input_dim = d_model, attention_dim = d_model)
        self.ffn1_weights = Linear(dim_in = d_model, dim_out = 128).forward()[0]
        self.ffn1_biases = Linear(dim_in = d_model, dim_out = 128).forward()[1]
        self.ffn2_weights = Linear(dim_in = 128, dim_out = d_model).forward()[0]
        self.ffn2_biases = Linear(dim_in = 128, dim_out = d_model).forward()[1]
        self.out_weights = Linear(dim_in = d_model, dim_out = 1).forward()[0]
    
    def forward(self, batch_input): #(x0)
        self.batch_input = batch_input
        x = batch_input @ self.input_embedding_matrix #(x1)
        x = x + self.pos_enc #(x2)
        
        ##transformer block##
        x_norm = self.layer_norm1.forward(x) #(x_norm1)
        attn_out = self.attention_network.forward(x_norm)
        x = x + attn_out #(x3)
        x_norm = self.layer_norm2.forward(x) #(x_norm2)
        self.x_norm2 = x_norm
        ffn1_out = relu((x_norm @ self.ffn1_weights + self.ffn1_biases))
        ffn2_out = ffn1_out @ self.ffn2_weights + self.ffn2_biases
        x = x + ffn2_out #(x4)
        x = x.mean(1) #(x5)
        self.x5 = x

        x = x @ self.out_weights #(x6)
        out = sigmoid(x) #y_pred

        self.ffn1_out_post_relu = ffn1_out
        self.x6 = x
        self.out = out

        return out
    
    def backward(self, input_gradient, learning_rate):
        x6_grad = input_gradient * (self.out) * (1-self.out) #(B,1)
        out_weight_gradient = self.x5.T @ x6_grad #(d_model, 1)
        x5_grad = x6_grad @ self.out_weights.T #(B,d_model)
        x4_grad = np.repeat(x5_grad[:, None, :], self.max_len, axis=1)/self.max_len #(B,max_len,d_model)

        ffn2_out_grad = x4_grad #(B, max_len, d_model)
        ffn2_weights_grad = np.einsum('bti,bto -> io', self.ffn1_out_post_relu, ffn2_out_grad) #(128, d_model) 
        ffn2_bias_grad = ffn2_out_grad.sum(axis=(0,1)) #(d_model,)

        ffn1_out_post_relu_grad = ffn2_out_grad @ self.ffn2_weights.T #(B,N,128)
        ffn1_out_pre_relu_grad = ffn1_out_post_relu_grad * (self.ffn1_out_post_relu > 0).astype(ffn1_out_post_relu_grad.dtype) #(B,N,128)
        ffn1_weights_grad = np.einsum('bti,bto-> io', self.x_norm2, ffn1_out_pre_relu_grad) #(d_model, 128)
        ffn1_bias_grad = ffn1_out_pre_relu_grad.sum(axis=(0,1)) #(128,)
        x_norm2_grad = ffn1_out_pre_relu_grad @ self.ffn1_weights.T #(B,N,d_model)
        
        layer_norm2_input_grad = self.layer_norm2.backward(x_norm2_grad,learning_rate) #(B,N,d_model)
        x3_grad = x4_grad + layer_norm2_input_grad #(B,N,d_model)
        attn_out_grad = x3_grad #(B,N,d_model)
        attn_in_grad = self.attention_network.backward(attn_out_grad, learning_rate) #(B,N,d_model)
        layer_norm1_input_grad = self.layer_norm1.backward(attn_in_grad, learning_rate) #(B,N,d_model)
        x2_grad = x3_grad + layer_norm1_input_grad #(B,N,d_model)
        x1_grad = x2_grad #(B,N,d_model)
        input_embedding_matrix_grad = np.einsum('bti,bto->io', self.batch_input, x1_grad)  #(max_res,d_model)
        batch_input_grad = x1_grad @ self.input_embedding_matrix.T #(B,N,max_res)

        self.input_embedding_matrix = self.input_embedding_matrix - input_embedding_matrix_grad * learning_rate
        self.out_weights = self.out_weights - out_weight_gradient * learning_rate
       
        self.ffn2_weights = self.ffn2_weights - ffn2_weights_grad * learning_rate
        self.ffn2_biases = self.ffn2_biases - ffn2_bias_grad * learning_rate
        self.ffn1_weights = self.ffn1_weights - ffn1_weights_grad * learning_rate
        self.ffn1_biases = self.ffn1_biases - ffn1_bias_grad * learning_rate

        return batch_input_grad

