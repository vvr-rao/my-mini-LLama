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

def get_tokenizer():
    file_name = f'./vocab/merges.pkl'
    with open(file_name, 'rb') as f:
        merges = pickle.load(f)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    return SimpleBPETokenizer(vocab)
