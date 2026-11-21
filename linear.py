import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = (np.random.randn(in_features, out_features) * 0.02).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)


    def forward(self, x: np.ndarray):
        self.x = x
        return x @ self.weight + self.bias


    def backward(self, grad_output: np.ndarray):
        x_2d = self.x.reshape(-1, self.x.shape[-1])
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        self.grad_weight += x_2d.T @ grad_2d
        axes_to_sum = tuple(range(grad_output.ndim - 1))
        self.grad_bias += np.sum(grad_output, axis=axes_to_sum)
        return grad_output @ self.weight.T


    def update(self, learning_rate: float):
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias
        self.grad_weight.fill(0)
        self.grad_bias.fill(0)