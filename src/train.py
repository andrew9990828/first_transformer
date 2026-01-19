# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Update: 1/19/26
# File Name: train.py
#
# Description:
#   This project implements a minimal single-head Transformer (self-attention)
#   model from scratch using pure NumPy.
#
#   The core ideas demonstrated are:
#     1) Token representations are projected into Queries, Keys, and Values,
#        enabling content-based attention over a sequence.
#     2) Positional information is injected via positional encodings so the
#        model can reason about token order.
#
#   The model is trained with the same fundamental learning loop used in a
#   standard neural network:
#     forward pass (Transformer -> logits),
#     loss computation (e.g., next-token cross-entropy),
#     backward pass (explicit gradients for every parameter),
#     optimizer step (parameter updates),
#     iteration over batches/epochs.
#
#   No machine learning frameworks are used. Every component—from attention
#   scores and softmax to LayerNorm, residuals, and parameter updates—is
#   implemented explicitly to reinforce a first-principles understanding of
#   how Transformers learn.





