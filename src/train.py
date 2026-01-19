# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Update: 1/19/26
# File Name: train.py
#
# Description:
#   This project implements a minimal single-head Transformer
#   (self-attention-based language model) from scratch using pure NumPy.
#
#   Instead of fabricated datasets, the model is trained directly on raw
#   text. A text file is tokenized into a sequence of token IDs, and
#   training examples are formed implicitly by predicting the next token
#   in a sequence (autoregressive language modeling).
#
#   The core ideas demonstrated are:
#     1) Queries, Keys, and Values enable content-based attention over tokens.
#     2) Positional encodings inject order information into token embeddings.
#
#   The model follows the same fundamental learning loop as any neural network:
#     forward pass (Transformer -> logits),
#     loss computation (next-token cross-entropy),
#     backward pass (explicit NumPy gradients),
#     optimizer step (parameter updates),
#     iteration over batches and epochs.
#
#   No machine learning frameworks are used. Every component—from tokenization
#   and attention to LayerNorm, residual connections, and gradient updates—is
#   implemented explicitly to reinforce a first-principles understanding of
#   how Transformers learn.
