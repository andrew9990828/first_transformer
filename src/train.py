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



# ==========================
# Setup stage(Steps 1-3)
# ==========================

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

# Build explicit character ↔ integer lookup tables.
# 
# - A dictionary is used (not a list) because we need to map
#   arbitrary characters (keys) → integer token IDs (values).
# - This acts as a static "translator": it does NOT compute anything,
#   it simply remembers the mapping discovered from the data.
# - enumerate(vocab) provides both the integer ID (idx) and the
#   corresponding character in a single loop.
# ****I'm still learning python thats for my reference :)****

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


# ===================================
# Step 4: Sample a batch of sequences
# ===================================


# Set B and T
B = 12
T = 8

# Randomize to be "i must be ≤ N - (T+1)""
N = len(encoded_txt)
max_start = N - (T+1)

inputs = []
targets = []

for _ in range(B):
    # Set random start point in the sequence of tokens
    i = np.random.randint(0, max_start + 1)

    x = encoded_txt[i : i+T]
    y = encoded_txt[i+1 : i+T+1]

    inputs.append(x)
    targets.append(y)

inputs = np.stack(inputs, axis=0)
targets = np.stack(targets, axis=0)


# ============================================
# Step 5: Token embedding & Postional encoding
# ============================================
#
# NOTE: So before I even attempted to translate this
# into actual code, I had to understand what a tensor was
# and also WHY we need token embeddings. I asked myself
# "Well all of the individual parts of the sequence
# are represented by a 1D vector of integers. Isn't that 
# good enough?"
# No, I was wrong, but this very curiosity REALLY made this click.
# That 1D stored vector is simply a set of IDs.
# We cannot just give the transformer and ID and say
# "Yeah here ya go man, predict the next one."
# So we have to convert those IDs to a tensor. A tensor
# is basically a 3D matrix and in our case specifically, 
# it would be (B, T, C) is B being the batches, T being the sequence,
# and then C being the depth of the tensor representing the token embedding.
#
# This is SO important because the transformer throughout this process is
# very simply using linear algebra to express attention later and update the embeddings
#

# Define sizes
V = len(vocab)
# C is the embedding dimension (I chose 32)
C = 32      

# Create the token emedding table
# Shape is (57, 32) meaning each unique element
# V=57 unique tokens, each gets a 32-dim vector.
token_embedding_table = np.random.uniform(-0.1, 0.1, size=(V, C))


# Now do the indexing with the batch we had
# Should create a tensor now with (B, T, C) as the shape.
tok = token_embedding_table[inputs]

# Create the postional embedding table with shape (T,C)
# Then add it to tok
positional_embedding_table = np.random.uniform(-0.1, 0.1, size=(T,C))
positions = np.arange(T)
pos = positional_embedding_table[positions]

# Now we combine the positional embeddings to
# the token embeddings
x_seq = tok + pos


# ========================================================================
# Step 6: Self-attention (single head: Q, K, V → attention → weighted sum)
# ========================================================================

# Define H
H = C

# Initalize weights
Wq = np.random.uniform(-0.1, 0.1, size=(C, H))
Wk = np.random.uniform(-0.1, 0.1, size=(C, H))
Wv = np.random.uniform(-0.1, 0.1, size=(C, H))

# Compute Q, K, and V
# We are computing Queries(Q), Keys(K), and Values(V)
# This is essential to understanding how transformers
# self-attention works:
#   Query (Q) → what am I looking for?
#   Key (K) → what do I contain?
#   Value (V) → what information do I offer?
Q = x_seq @ Wq
K = x_seq @ Wk
V = x_seq @ Wv

#Check shapes
print(Q.shape, K.shape, V.shape)


# Now THIS is where the self-attention math starts.
# We need to give each token the ability to compare itself
# to every other token AKA scoring.
# ========================================================
# THE KEY IDEA: Each token produces 8 scores — one score
# for every token in the sequence, including itself.
# ========================================================

# ========================================================
# 1) Transpose K so dot products are possible
# ========================================================
# K has shape: (B, T, H)
# We want each query (length H) to dot with every key (length H)
#
# So we transpose ONLY the last two dimensions:
#   (B, T, H) -> (B, H, T)
#
# This allows:
#   (B, T, H) @ (B, H, T) -> (B, T, T)
# which gives pairwise token-to-token scores
K_T = np.transpose(K, (0, 2, 1))


# ========================================================
# 2) Compute raw attention scores (dot products)
# ========================================================
# This performs batch-wise matrix multiplication.
#
# For each batch b:
#   scores[b] = Q[b] @ K[b].T
#
# Meaning:
#   scores[b, i, j] = dot(Q[b, i], K[b, j])
#
# Interpretation:
#   "How relevant is token j to token i?"
#
# Shape:
#   Q    : (B, T, H)
#   K_T  : (B, H, T)
#   scores -> (B, T, T)
scores = Q @ K_T


# ========================================================
# 3) Scale the scores for numerical stability
# ========================================================
# As H grows, dot products grow in magnitude.
# This makes softmax extremely sharp and kills gradients.
#
# Dividing by sqrt(H) keeps values in a reasonable range.
# This does NOT change the meaning of attention,
# it just makes training stable.
scores = scores / np.sqrt(H)

