# nano_transformers

Project: Nano-Transformers

Objective: Implement a lightweight, modular transformer library in pure Python (with optional PyTorch for tensor ops) that covers core models (e.g., BERT, GPT) for educational purposes. No external dependencies beyond PyTorch (optional). Focus on readability, minimalism, and understanding.

Step 1: Define Scope

Let’s start small and build iteratively. Core components to implement:

* Multi-Head Self-Attention: The heart of transformers.
* Feed-Forward Network (FFN): Position-wise dense layers.
* Positional Encoding: For sequence order.
* Layer Normalization: For stable training.
* Transformer Block: Combines attention and FFN.
* Models:
    + Encoder-only (BERT-like) for classification tasks.
    + Decoder-only (GPT-like) for generation.
    + Optional: Encoder-Decoder (T5-like) later.
    + Other models variants(dense and sparse models e.g., MoE)

We’ll aim for a simple use case first, like a toy language model or text classification, to test the components.
