from collections import Counter


class Tokenizer():
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []


    def merge_symbols(self, symbols: str, pair: list):
        new_symbols = []
        i = 0
        while i < len(symbols):
            if (i < len(symbols) - 1
            and symbols[i] == pair[0]
            and symbols[i + 1] == pair[1]):
                new_symbols.append(pair[0] + pair[1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return tuple(new_symbols)


    def train(self, text: str):
        words = text.split()
        word_freqs = Counter(words)
        splits = {word: (*list(word), "</>") for word in word_freqs.keys()}
        vocab = set()

        for symbols in splits.values():
            vocab.update(symbols)
        num_merges = self.vocab_size - len(vocab)

        for _ in range(num_merges):
            pairs = Counter()
            for word, freq in word_freqs.items():
                symbols = splits[word]
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j + 1])
                    pairs[pair] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)

            new_splits = {}
            for word in word_freqs.keys():
                new_splits[word] = self.merge_symbols(splits[word], best_pair)
            splits = new_splits

            self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
                
    
    def encode(self, text: str):
        words = text.split()
        result = []
        for word in words:
            symbols = (*list(word), "</>")
            for pair in self.merges:
                symbols = self.merge_symbols(symbols, pair)
            result.extend(self.vocab[token] for token in symbols)
        return result


    def decode(self, tokens: list[int]) -> str:
        inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        result = []
        for token_id in tokens:
            token = inverse_vocab[token_id]

            if token == "</>":
                result.append(" ")
            else:
                result.append(token)

        return "".join(result).strip()


def split_tokens(tokens, val_fraction: float, seq_len: int):
    if not val_fraction:
        return tokens, None
    n_val = int(len(tokens) * val_fraction)
    if n_val < seq_len + 1:
        raise ValueError(
            f"Для валидации нужно хотя бы {seq_len + 1} токенов, а при VAL_FRACTION={val_fraction} "
            f"их {n_val} (всего токенов: {len(tokens)}). Дай больше текста или увеличь VAL_FRACTION.")
    return tokens[:-n_val], tokens[-n_val:]