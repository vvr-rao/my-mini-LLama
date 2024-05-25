import pickle

class SimpleBPETokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_len = len(merges)

    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
          stats = get_stats(tokens)
          pair = min(stats, key=lambda p: merges.get(p, float("inf")))
          if pair not in merges:
            break # nothing else can be merged
          idx = merges[pair]
          tokens = merge(tokens, pair, idx)
        return tokens

    def decode(ids):
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

def load_vocab(size: int):
    file_name = f'./vocab/vocab.pkl'
    with open(file_name, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
