import numpy as np
from .feedforward import FeedForward
from .layernorm import LayerNorm
from .mha import MultiHeadAttention

class Decoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.blocks = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]


    def forward(self, x) -> np.ndarray:
        for block in self.blocks:
            x = block.forward(x)
        return x


    def backward(self, grad_output) -> np.ndarray:
        grad = grad_output
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        return grad


    def update(self, learning_rate):
        for block in self.blocks:
            block.update(learning_rate)


class DecoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)


    def forward(self, x: np.ndarray):
        norm_x = self.norm1.forward(x)
        attention_out = self.attention.forward(norm_x)
        x = x + attention_out

        norm_x = self.norm2.forward(x)
        ff_out = self.feed_forward.forward(norm_x)
        x = x + ff_out

        return x
    

    def backward(self, grad_output):
        grad_ff = self.feed_forward.backward(grad_output)
        grad_h = grad_output + self.norm2.backward(grad_ff)
        grad_attn = self.attention.backward(grad_h)
        return grad_h + self.norm1.backward(grad_attn)
    

    def update(self, learning_rate: float):
        self.attention.update(learning_rate)
        self.feed_forward.update(learning_rate)

        self.norm1.update(learning_rate)
        self.norm2.update(learning_rate)