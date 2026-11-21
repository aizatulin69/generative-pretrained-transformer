import numpy as np
from math import sqrt, pi
from .linear import Linear

_GELU_C = sqrt(2.0 / pi)
_GELU_A = 0.044715

class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: np.ndarray):
        self.hidden = self.linear1.forward(x)
        h = self.hidden
        self.tanh_u = np.tanh(_GELU_C * (h + _GELU_A * h ** 3))
        self.activated = 0.5 * h * (1.0 + self.tanh_u)
        return self.linear2.forward(self.activated)


    def backward(self, grad_output: np.ndarray):
        grad = self.linear2.backward(grad_output)
        h, t = self.hidden, self.tanh_u
        d_gelu = 0.5 * (1.0 + t) + 0.5 * h * (1.0 - t*t) * _GELU_C * (1.0 + 3.0 * _GELU_A * h*h)
        grad = grad * d_gelu
        return self.linear1.backward(grad)


    def update(self, learning_rate: float):
        self.linear1.update(learning_rate)
        self.linear2.update(learning_rate)
