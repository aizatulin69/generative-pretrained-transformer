import numpy as np


class EmbeddingBlock:
    def __init__(self, vocab_size: int, max_seq_len:int, dim):
        self.TOK_embedding = Embedding(vocab_size, dim)
        self.POS_embedding = Embedding(max_seq_len, dim)
        self.scale = np.float32(np.sqrt(dim))


    def forward(self, tokens: np.ndarray):
        _, seq, = tokens.shape
        positions = np.arange(seq)
        tok_emb = self.TOK_embedding.forward(tokens)
        pos_emb = self.POS_embedding.forward(positions)
        return (tok_emb + pos_emb) * self.scale


    def backward(self, grad_output: np.ndarray):
        grad_output = grad_output * self.scale
        self.TOK_embedding.backward(grad_output)
        self.POS_embedding.backward(grad_output.sum(axis=0))


    def update(self, learning_rate: float):
        self.TOK_embedding.update(learning_rate)
        self.POS_embedding.update(learning_rate)


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.weight = (np.random.randn(vocab_size, embedding_dim) * 0.02).astype(np.float32)
        self.grad = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        self.tokens = None


    def forward(self, indices: np.ndarray) -> np.ndarray[np.float32]:
        indices = indices.astype(np.int64)
        self.tokens = indices.copy()
        return self.weight[indices]


    def backward(self, grad_output: np.ndarray):
        grad_output = grad_output.astype(np.float32)
        flat_tokens = self.tokens.ravel()
        flat_grads = grad_output.reshape(-1, self.grad.shape[1])
        np.add.at(self.grad, flat_tokens, flat_grads)


    def update(self, learning_rate: float):
        self.weight -= learning_rate * self.grad
        self.grad.fill(0)
