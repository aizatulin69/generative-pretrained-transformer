import numpy as np
from math import cos, pi


def collect_params(*objects):
    pairs_names = (("weight", "grad_weight"), ("weight", "grad"), ("bias", "grad_bias"),
    ("gamma", "grad_gamma"), ("beta", "grad_beta"))
    out, seen = [], set()

    def walk(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, (list, tuple)):
            for o in obj:
                walk(o)
        elif hasattr(obj, "__dict__"):
            d = vars(obj)
            for pn, gn in pairs_names:
                if pn in d and gn in d:
                    out.append((d[pn], d[gn]))
            for v in d.values():
                if not isinstance(v, np.ndarray):
                    walk(v)

    for o in objects:
        walk(o)
    return out


def count_parameters(*objects):
    visited = set()
    total = 0
    parameter_names = {"weight", "bias", "gamma", "beta", "w", "b",
    "wq", "wk", "wv", "wo", "w1", "w2", "w3", "b1", "b2", "b3"}

    def count(obj):
        nonlocal total
        if id(obj) in visited:
            return
        visited.add(id(obj))
        if isinstance(obj, (list, tuple)):
            for item in obj:
                count(item)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in parameter_names:
                    if isinstance(value, np.ndarray):
                        total += value.size
                else:
                    count(value)
            return
        if hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                if key.lower() in parameter_names:
                    if isinstance(value, np.ndarray):
                        total += value.size
                else:
                    count(value)
                    
    for obj in objects:
        count(obj)
    return total


def lr_at(step: int, total_steps: int, lr: float, warmup_steps: int, min_ratio: float) -> float:
    if warmup_steps and step <= warmup_steps:
        return lr * step / warmup_steps
    if min_ratio >= 1.0:
        return lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + cos(pi * min(1.0, progress))))


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, clip=1.0):
        self.params, self.lr, self.eps, self.clip = params, lr, eps, clip
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0


    def step(self) -> float:
        self.t += 1
        norm = np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for _, g in self.params))
        scale = min(1.0, self.clip / (norm + 1e-6)) if self.clip else 1.0
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for (p, g), m, v in zip(self.params, self.m, self.v):
            g = g * scale
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g*g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
        for _, g in self.params:
            g.fill(0)
        return norm