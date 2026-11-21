import numpy as np
from .linear import Linear


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
    

    def forward(self, x: np.ndarray):
        batch, seq, _ = x.shape

        self.Q_linear = self.W_Q.forward(x)
        self.K_linear = self.W_K.forward(x)
        self.V_linear = self.W_V.forward(x)

        Q = self.Q_linear.reshape(batch, seq, self.num_heads, self.head_dim)
        K = self.K_linear.reshape(batch, seq, self.num_heads, self.head_dim)
        V = self.V_linear.reshape(batch, seq, self.num_heads, self.head_dim)

        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        self.Q, self.K, self.V = Q, K, V

        out, self.attention_weights = self.attention(Q, K, V)

        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(batch, seq, self.d_model)

        return self.W_O.forward(out)


    def backward(self, grad_output: np.ndarray[np.float32]):
        batch, seq, _ = grad_output.shape

        grad = self.W_O.backward(grad_output)

        grad = grad.reshape(batch, seq, self.num_heads, self.head_dim)
        grad = grad.transpose(0, 2, 1, 3)

        grad_Q, grad_K, grad_V = self.attention_backward(grad)

        grad_Q = grad_Q.transpose(0, 2, 1, 3)
        grad_K = grad_K.transpose(0, 2, 1, 3)
        grad_V = grad_V.transpose(0, 2, 1, 3)

        grad_Q = grad_Q.reshape(batch, seq, self.d_model)
        grad_K = grad_K.reshape(batch, seq, self.d_model)
        grad_V = grad_V.reshape(batch, seq, self.d_model)

        grad_xQ = self.W_Q.backward(grad_Q)
        grad_xK = self.W_K.backward(grad_K)
        grad_xV = self.W_V.backward(grad_V)

        return grad_xQ + grad_xK + grad_xV
        

    def update(self, learning_rate):
        self.W_Q.update(learning_rate)
        self.W_K.update(learning_rate)
        self.W_V.update(learning_rate)
        self.W_O.update(learning_rate)
    
    
    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, dropout=None):
        d_k = K.shape[-1]
        K_transposed = K.transpose(0, 1, 3, 2)
        scores = (Q @ K_transposed) / (d_k**0.5)
        causal_mask = np.triu(np.ones(scores.shape[-2:], dtype=bool),k=1)        
        scores = np.where(causal_mask, -1e9, scores)
        attention_weights = self.softmax(scores, axis=-1)
        if dropout is not None: 
            attention_weights = dropout(attention_weights)
        return attention_weights @ V, attention_weights
    
    def softmax(self, x: np.ndarray, axis=-1) -> np.ndarray[np.float32]:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    

    def attention_backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_k = self.K.shape[-1]
        sqrt_d_k = d_k**0.5
        grad_V = self.attention_weights.transpose(0, 1, 3, 2) @ grad_output
        grad_A = grad_output @ self.V.transpose(0, 1, 3, 2)
        grad_S = self.softmax_backward(grad_A, self.attention_weights)
        grad_Q = grad_S @ self.K / sqrt_d_k
        grad_K = grad_S.transpose(0, 1, 3, 2) @ self.Q / sqrt_d_k
        return grad_Q, grad_K, grad_V
    
    def softmax_backward(self, grad_A, A) -> np.ndarray:
        sum_term = np.sum(grad_A * A, axis=-1, keepdims=True)
        grad_S = A * (grad_A - sum_term)
        return grad_S