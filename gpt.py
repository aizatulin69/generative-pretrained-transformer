import numpy as np
import matplotlib.pyplot as plt

from .tokenizer import Tokenizer, split_tokens
from .embedding import EmbeddingBlock
from .decoder import Decoder
from .layernorm import LayerNorm
from .linear import Linear
from .adam import collect_params, count_parameters, lr_at, Adam

from .nnstuff import count_windows, cross_entropy, create_batches, evaluate

TEXT = None
PATH = r"E:\Git\PROJECTS\transformer\transformer\text.txt"
VOCAB_SIZE = 1000
MAX_SEQ_LEN = 64
D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
NUM_LAYERS = 8
BATCH_SIZE = 16
EPOCH = 10
LR = 5e-4
VAL_FRACTION = 0.1
STRIDE = 1
WARMUP_STEPS = 0
MIN_LR_RATIO = 1.0
SEED = 67
SAVE_PLOT = False
HISTORY_STEPS = 8


class GPT():
    def __init__(self, TEXT=TEXT, PATH=PATH, VOCAB_SIZE=VOCAB_SIZE, MAX_SEQ_LEN=MAX_SEQ_LEN, D_MODEL=D_MODEL, NUM_HEADS=NUM_HEADS, D_FF=D_FF, NUM_LAYERS=NUM_LAYERS, BATCH_SIZE=BATCH_SIZE, EPOCH=EPOCH, LR=LR, VAL_FRACTION=VAL_FRACTION, STRIDE=STRIDE, WARMUP_STEPS=WARMUP_STEPS, MIN_LR_RATIO=MIN_LR_RATIO, SEED=SEED, SAVE_PLOT=SAVE_PLOT, HISTORY_STEPS=HISTORY_STEPS):
        self.TEXT = TEXT
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.D_MODEL = D_MODEL
        self.NUM_HEADS = NUM_HEADS
        self.D_FF = D_FF
        self.NUM_LAYERS = NUM_LAYERS
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCH = EPOCH
        self.LR = LR
        self.VAL_FRACTION = VAL_FRACTION
        self.STRIDE = STRIDE
        self.WARMUP_STEPS = WARMUP_STEPS
        self.MIN_LR_RATIO = MIN_LR_RATIO
        self.SEED = SEED
        self.SAVE_PLOT = SAVE_PLOT
        self.HISTORY_STEPS = HISTORY_STEPS

        if self.TEXT == None:
            with open(PATH, "r", encoding="utf-8") as t:
                self.TEXT = t.read()

        self.vocab = Tokenizer(vocab_size=self.VOCAB_SIZE)
        self.vocab.train(self.TEXT)
        tokens = self.vocab.encode(self.TEXT)

        self.embedding = EmbeddingBlock(vocab_size=self.VOCAB_SIZE, max_seq_len=self.MAX_SEQ_LEN, dim=self.D_MODEL)
        self.layernorm = LayerNorm(self.D_MODEL)
        self.decoder = Decoder(self.D_MODEL, self.NUM_HEADS, self.D_FF, self.NUM_LAYERS)
        self.linear = Linear(self.D_MODEL, self.VOCAB_SIZE)

        optimizer_params = collect_params(self.embedding, self.decoder, self.layernorm, self.linear)
        self.params = count_parameters(self.embedding, self.decoder, self.layernorm, self.linear)
        self.optimizer = Adam(optimizer_params, lr=self.LR, clip=None)

        self.history = {"grad_norm": [],
            "train_loss": [], "train_ppl": [], "train_acc": [],
            "val_loss": [], "val_ppl": [], "val_acc": []}

        self.step = 0

        np.random.seed(self.SEED)
        self.rng = np.random.default_rng(self.SEED)

        self.train_tokens, self.val_tokens = split_tokens(tokens, self.VAL_FRACTION, self.MAX_SEQ_LEN)

        n_train_windows = count_windows(len(self.train_tokens), self.MAX_SEQ_LEN, self.STRIDE)

        if n_train_windows == 0:
            raise ValueError(
                f"Слишком мало токенов для train ({len(self.train_tokens)}): "
                f"нужно хотя бы {self.MAX_SEQ_LEN + 1}")

        if n_train_windows < self.BATCH_SIZE:
            print(
                f"ВНИМАНИЕ: в train всего {n_train_windows} окон "
                f"< BATCH_SIZE={self.BATCH_SIZE}; размер батча уменьшен "
                f"до {n_train_windows}")
            self.BATCH_SIZE = n_train_windows

        batches_per_epoch = n_train_windows // self.BATCH_SIZE
        self.total_steps = batches_per_epoch * self.EPOCH

        self.val_batches = None

        if self.val_tokens is not None:
            self.val_batches = create_batches(
                self.val_tokens,
                self.MAX_SEQ_LEN,
                self.BATCH_SIZE,
                drop_last=False
            )

        print(f"Text length:  {len(self.TEXT)}")
        print(f"Text tokens:  {len(tokens)}  (unique ids used: {len(set(tokens))} of {self.VOCAB_SIZE})")
        print(f"Train tokens: {len(self.train_tokens)}")
        print(f"Val tokens:   {0 if self.val_tokens is None else len(self.val_tokens)}")

    
    def _train_step(self, x, y):
        emb_out = self.embedding.forward(x)
        dec_out = self.decoder.forward(emb_out)
        norm_out = self.layernorm.forward(dec_out)
        logits = self.linear.forward(norm_out)

        loss, accuracy, grad = cross_entropy(logits, y)

        grad = self.linear.backward(grad)
        grad = self.layernorm.backward(grad)
        grad = self.decoder.backward(grad)
        self.embedding.backward(grad)

        grad_norm = self.optimizer.step()

        return loss, accuracy, grad_norm
    

    def train(self):
        loss_window = []
        acc_window = []
        grad_window = []

        self._init_plot()

        for e in range(self.EPOCH):
            epoch_losses = []
            epoch_accuracies = []
            epoch_grad_norms = []

            batches = create_batches(
                self.train_tokens,
                self.MAX_SEQ_LEN,
                self.BATCH_SIZE,
                self.STRIDE,
                drop_last=True,
                rng=self.rng
            )

            for x, y in batches:
                self.step += 1

                self.optimizer.lr = lr_at(
                    self.step,
                    self.total_steps,
                    self.LR,
                    self.WARMUP_STEPS,
                    self.MIN_LR_RATIO
                )

                loss, accuracy, grad_norm = self._train_step(x, y)

                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
                epoch_grad_norms.append(grad_norm)

                loss_window.append(loss)
                acc_window.append(accuracy)
                grad_window.append(grad_norm)

                if len(loss_window) > self.HISTORY_STEPS:
                    loss_window.pop(0)
                    acc_window.pop(0)
                    grad_window.pop(0)

                avg_loss = np.mean(loss_window)
                avg_acc = np.mean(acc_window)
                avg_grad = np.mean(grad_window)
                ppl = np.exp(avg_loss)

                self.history["train_loss"].append(avg_loss)
                self.history["train_ppl"].append(ppl)
                self.history["train_acc"].append(avg_acc)
                self.history["grad_norm"].append(avg_grad)

                self._update_plot()

            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            perplexity = np.exp(avg_loss)

            line = (
                f"{e + 1:03d}/{self.EPOCH} | "
                f"Loss: {avg_loss:.5f} | "
                f"PPL: {perplexity:.3f} | "
                f"Accuracy: {avg_accuracy:.4f} | "
            )

            if self.val_batches is not None:
                val_loss, val_acc, val_ppl = evaluate(
                    self.val_batches,
                    self.embedding,
                    self.decoder,
                    self.layernorm,
                    self.linear
                )

                self.history["val_loss"].append(val_loss)
                self.history["val_ppl"].append(val_ppl)
                self.history["val_acc"].append(val_acc)

                line += (
                    f"Val Loss: {val_loss:.5f} | "
                    f"Val PPL: {val_ppl:.3f} | "
                    f"Val Acc: {val_acc:.4f} | "
                )

            line += f"|g|: {np.mean(epoch_grad_norms):.3f}"
            print(line)

        print("\n========== TRAINING RESULTS ==========")
        print(f"Final Train Loss:       {avg_loss:.6f}")
        print(f"Final Train Perplexity: {perplexity:.4f}")
        print(f"Final Train Accuracy:   {avg_accuracy:.4f}")

        if self.val_batches is not None:
            best = int(np.argmin(self.history["val_loss"]))

            print(f"Final Val Loss:         {self.history['val_loss'][-1]:.6f}")
            print(f"Final Val Perplexity:   {self.history['val_ppl'][-1]:.4f}")
            print(f"Final Val Accuracy:     {self.history['val_acc'][-1]:.4f}")
            print(f"Best Val Loss:          {self.history['val_loss'][best]:.6f}  (epoch {best + 1})")

        print(f"Parameters:       {self.params:,}")
        print("=====================================\n")

        return self.history
    

    def _init_plot(self):
        plt.ion()

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 4))
        self.lines = {}

        for ax, key, title in (
            (self.axes[0], "loss", "Cross Entropy Loss"),
            (self.axes[1], "acc", "Token Accuracy")):

            self.lines[f"train_{key}"], = ax.plot([], [], label="Train")
            self.lines[f"val_{key}"], = ax.plot([], [], label="Validation")

            ax.set_title(title)
            ax.set_xlabel("Training steps")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()

        self.lines["grad_norm"], = self.axes[2].plot([], [], label="Gradient Norm", color="brown")

        self.axes[2].set_title("Gradient Norm")
        self.axes[2].set_xlabel("Training steps")
        self.axes[2].grid(True, linestyle=":", alpha=0.6)
        self.axes[2].legend()

        plt.tight_layout()
        plt.show(block=False)

    
    def _update_plot(self):
        train_steps = np.arange(1, len(self.history["train_loss"]) + 1)
        batches_per_epoch = self.total_steps // self.EPOCH
        val_steps = np.arange(1, len(self.history["val_loss"]) + 1) * batches_per_epoch

        for i, key in enumerate(("loss", "acc")):
            self.lines[f"train_{key}"].set_data(train_steps, self.history[f"train_{key}"])
            self.lines[f"val_{key}"].set_data(val_steps, self.history[f"val_{key}"])
            self.axes[i].relim()
            self.axes[i].autoscale_view()

        for step in val_steps:
            for ax in self.axes:
                ax.axvline(step, color="green", linestyle="--", alpha=0.5)

        self.lines["grad_norm"].set_data(train_steps, self.history["grad_norm"])
        self.axes[2].relim()
        self.axes[2].autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events() 
        plt.pause(0.001)