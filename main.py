# Example usage script
import torch
from models.gpt.model import GPT
from utils.tokenizer import SimpleTokenizer

# A simple example vocabulary
vocab = ['<pad>', 'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd']
tokenizer = SimpleTokenizer(vocab)

# Model parameters
vocab_size = len(vocab)
d_model = 64
d_ff = 128
num_layers = 2
num_heads = 2
max_len = 10
# Instantiate the model
model = GPT(vocab_size, d_model, num_heads, d_ff, num_layers)

# Example input: "hello worl"
text = "hello wor"
tokens = tokenizer.encode(text)
input_tensor = torch.tensor(tokens).unsqueeze(0)

# Forward pass
with torch.no_grad():
    output = model(input_tensor)

# Get the predicted token for next word
predicted_token_index = torch.argmax(output[0, -1, :])
predicted_char = tokenizer.decode([predicted_token_index.item()])

print(f"Input: '{text}'")
print(f"Predicted next character: '{predicted_char}'")