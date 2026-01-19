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


# =========================
# Transformer Training Guide (Pure NumPy)
# =========================
#
# SETUP (run once)
# 1. Read raw text file
# 2. Build vocabulary (char → int, int → char)
# 3. Encode entire text as a 1D array of token IDs
#
# TRAINING LOOP
# Outer loop: for epoch in epochs
#   Inner loop: for step in steps_per_epoch
#
# 4. Sample a batch of sequences
#    - tokens  : (B, T)
#    - targets : (B, T)  # next-token prediction (shifted by +1)
#
# FORWARD PASS (Transformer)
# 5. Token embedding + positional encoding
# 6. Self-attention (single head: Q, K, V → attention → weighted sum)
# 7. Residual connection + LayerNorm
# 8. Feed-forward network (MLP)
# 9. Residual connection + LayerNorm
# 10. Output projection → logits (B, T, V)
#
# LEARNING
# 11. Cross-entropy loss (logits vs targets)
# 12. Backpropagation (explicit NumPy gradients)
# 13. Optimizer step (SGD / Adam)
#
# 14. Repeat for all batches → next epoch

# ==========================
# Define the problem/goal
# ==========================
# Given a sequence of tokens from a text file, predict the next
# token at each position using a single-head self-attention
# Transformer trained from scratch in NumPy.

import numpy as np
import random

# -------------------------
# Reproducibility
# -------------------------
# Fixing the random seeds ensures that:
# - The synthetic dataset is generated identically on every run
# - Weight initialization starts from the same parameters
# - Training dynamics (loss and accuracy curves) are repeatable
#
# This is critical for debugging, validating learning behavior,
# and demonstrating that observed improvements come from the
# optimization process itself—not from random chance.

np.random.seed(42)
random.seed(42)

# First we just need to read the file and just store it as one large string
with open('data_textfiles/the_raven_edgarallanpoe.txt', 'r', encoding="utf-8") as file:
    content = file.read()

# print(content) just tested we read the file right

# Now we need to loop through the content and find each unique
# char in the entire textfile and store those as an array of
# 1d integers

# Use pythons set function to store unique elements automatically
unique_chars = set()

for ch in content:
    # Store all unique elements
    unique_chars.add(ch)

# Find the length of the list of unique elements
num_of_chars = len(unique_chars)

# Store the chars as ordered using pythons sorted() function
vocab = sorted(unique_chars)

# Create mappings
char_to_idx = {}
idx_to_char = {}

# Loop through to map everything
for idx, char in enumerate(vocab):
    char_to_idx[char] = idx
    idx_to_char[idx] = char

encoded = []
for ch in content:
    encoded.append(char_to_idx[ch])

encoded_txt = np.array(encoded, dtype=int)

print(encoded_txt)




    


