# GPT from scratch in NumPy

A tiny GPT-style language model written with **only NumPy**, including the backward pass.
No PyTorch, no TensorFlow, no autograd: every gradient is derived and coded by hand.

## What it does

It predicts the **next token**. Given a piece of text, the model guesses what comes next,
compares the guess with the real text, and adjusts its weights to guess better next time.
Repeat this many times and it learns the patterns of the text.

## How it works

1. **Tokenizer (BPE).** Text is split into tokens: frequent words and word pieces get their own number.
2. **Embeddings.** Each token number becomes a vector. A second vector for the position tells the model the word order.
3. **Decoder blocks (x8).** Each block has two parts:
   - *Attention*: every token looks at the tokens before it (never at the future) and decides which ones matter.
   - *Feed-forward network*: a small neural net that processes each token on its own.
4. **Output layer.** The final vector is turned into one score per vocabulary token, then into probabilities with softmax.
5. **Learning.** The loss (cross-entropy) measures how wrong the guess was. The backward pass computes how each weight should change, and the Adam optimizer applies the change.

## Architecture

```
token ids
  -> token embedding + position embedding   (scaled by sqrt(d_model))
  -> 8 x Decoder block:
         x = x + Attention(LayerNorm(x))     causal multi-head self-attention
         x = x + FeedForward(LayerNorm(x))   Linear -> GELU -> Linear
  -> LayerNorm
  -> Linear (d_model -> vocab size)
  -> softmax -> probabilities of the next token
```

This is the "Pre-LN" design used in GPT-2: LayerNorm goes *before* each sub-layer, and the
residual connection (`x = x + ...`) lets gradients flow straight through the whole network.

Default settings (`train.py`):

| Setting | Value |
|---|---|
| Vocabulary size | 1000 |
| Context length | 64 tokens |
| Model width (`d_model`) | 128 |
| Attention heads | 4 |
| Feed-forward width | 512 |
| Layers | 8 |
| Batch size | 16 |
| Parameters | about 1.85 million |

## Technologies

- **Python + NumPy**: all the math (matrix multiplication, softmax, LayerNorm, attention, gradients).
- **Matplotlib**: training curves.
- **Written from scratch**: BPE tokenizer, Adam optimizer with gradient clipping, learning-rate warmup and cosine decay.

## Files

| File | What it contains |
|---|---|
| `tokenizer.py` | BPE tokenizer and the train/validation split of tokens |
| `embedding.py` | Token and position embeddings |
| `linear.py` | Fully connected layer |
| `layernorm.py` | Layer normalization |
| `mha.py` | Causal multi-head attention |
| `feedforward.py` | Feed-forward network with GELU |
| `decoder.py` | Decoder block and the stack of blocks |
| `adam.py` | Adam optimizer, parameter collection, learning-rate schedule |
| `nnstuff.py` | Cross-entropy loss, batching, evaluation |
| `train.py` | Training loop, validation, plots |
| `gradcheck.py` | Checks the hand-written gradients against numerical ones |
| `bench.py` | Speed benchmark of one training step |

## How training works

- The text is cut into windows of 64 tokens. The target is the same window shifted by one token.
- `STRIDE` sets how much windows overlap. `STRIDE = 1` gives many more training steps per epoch.
- The last 10% of the text is kept for validation. Reported metrics: loss, perplexity and token accuracy.
- Every layer has a `forward` (computes the output, remembers what it needs) and a `backward` (uses the chain rule to pass the gradient back).
- `gradcheck.py` compares these gradients with finite differences, so the backward pass is tested.

## Run

```
pip install numpy matplotlib
```

Put your text in `text.txt`, then:

```
python train.py
```

Settings are the constants at the top of `train.py`.

## Notes

- With a very small text the model memorizes it: training accuracy gets high, validation stays poor. More text is the real fix.
- NumPy runs on the CPU only, so it is much slower than PyTorch on a GPU. This project is for learning how GPT works inside.
