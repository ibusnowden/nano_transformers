# A basic world level tokenizer

class SimpleTokenizer:
    def __init__(self, corpus):
        self.vocab = sorted(set("".join(corpus).split()))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text):
        return [self.word2idx.get(w, 0) for w in text.split()]
    
    def decode(self, indices):
        return " ".join(self.idx2word.get(i, "<UNK>") for i in indices)
