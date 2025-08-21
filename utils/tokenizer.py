# A simple tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {s:i for i, s in enumerate(vocab)}
        self.itos = {i:s for s, i in enumerate(vocab)}

    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):
        return "".join([self.itos[i] for i in tokens])