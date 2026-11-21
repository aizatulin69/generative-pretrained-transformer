import numpy as np


def count_windows(num_tokens: int, seq_len: int, stride=None) -> int:
    return len(range(0, num_tokens - seq_len, stride or seq_len))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float, np.ndarray]:
    batch, seq, vocab_size = logits.shape

    logits_2d = logits.reshape(-1, vocab_size)
    targets_1d = targets.reshape(-1)

    shifted = logits_2d - np.max(logits_2d, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    probs = np.exp(log_probs)

    n = len(targets_1d)
    loss = -np.mean(log_probs[np.arange(n), targets_1d])
    accuracy = np.mean(np.argmax(logits_2d, axis=-1) == targets_1d)

    grad = probs.copy()
    grad[np.arange(n), targets_1d] -= 1
    grad /= n
    
    return loss, accuracy, grad.reshape(batch, seq, vocab_size)


def evaluate(batches, embedding, decoder, layernorm, linear):
    total_loss = total_correct = total_tokens = 0.0

    for x, y in batches:
        emb_out = embedding.forward(x)
        dec_out = decoder.forward(emb_out)
        norm_out = layernorm.forward(dec_out)
        logits = linear.forward(norm_out)

        loss, accuracy, _ = cross_entropy(logits, y)

        n = y.size
        total_loss += loss * n
        total_correct += accuracy * n
        total_tokens += n

    avg_loss = total_loss / total_tokens

    return avg_loss, total_correct / total_tokens, float(np.exp(avg_loss))


def create_batches(tokens, seq_len: int, batch_size: int, stride=None, drop_last=True, rng=None) -> list:
    tokens = np.asarray(tokens, dtype=np.int64)
    starts = np.arange(0, len(tokens) - seq_len, stride or seq_len)
    if len(starts) == 0:
        raise ValueError(f"Слишком мало токенов ({len(tokens)}): нужно хотя бы {seq_len + 1}")
    if rng is not None:
        starts = rng.permutation(starts)
    batches = []
    for i in range(0, len(starts), batch_size):
        s = starts[i:i + batch_size]
        if drop_last and len(s) < batch_size:
            break
        x = np.stack([tokens[j:j + seq_len] for j in s])
        y = np.stack([tokens[j + 1:j + seq_len + 1] for j in s])
        batches.append((x, y))
    return batches