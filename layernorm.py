import numpy as np


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self.beta = np.zeros(normalized_shape, dtype=np.float32)

        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)


    def forward(self, x: np.ndarray) -> np.ndarray[np.float32]:
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        out = self.gamma * self.x_hat + self.beta
        return out
    
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray[np.float32]:
        N = self.x.shape[-1]
        axes_to_sum = tuple(range(grad_output.ndim - 1))
        self.grad_gamma += np.sum(grad_output * self.x_hat, axis=axes_to_sum)
        self.grad_beta += np.sum(grad_output, axis=axes_to_sum)
        inv_std = 1.0 / np.sqrt(self.var + self.eps)
        dx_hat = grad_output * self.gamma

        grad_x = (inv_std / N * 
            (N * dx_hat 
             - np.sum(dx_hat, axis=-1, keepdims=True) 
             - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)))

        return grad_x
    

    def update(self, learning_rate: float):
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta
        self.grad_gamma.fill(0)
        self.grad_beta.fill(0)