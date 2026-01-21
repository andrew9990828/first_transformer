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


# In language modeling, a token must NOT be allowed to attend
# to future tokens in the sequence.
#
# For a sequence of length T:
#   - token i may attend to tokens [0 .. i]
#   - token i must NOT attend to tokens [i+1 .. T-1]
#
# We enforce this by creating an upper-triangular mask where:
#   True  = future position (BLOCK attention)
#   False = allowed position
#
# These masked positions will later be set to a very large
# negative value so that, after softmax, their attention
# weight becomes effectively zero.
causal_mask = np.triu(np.ones((T, T)), k=1).astype(bool)
scores_masked = scores.copy()

# causal_mask is (T, T). This line broadcasts over batch dimension (B)
scores_masked[:, causal_mask] = -1e9   # effectively -inf for softmax

# Softmax calculated here. Inspired from my NN froms scratch
# This is the softmax over attention scores
scores_max = np.max(scores_masked, axis=-1, keepdims=True)
scores_exp = np.exp(scores_masked - scores_max)
sum_exp = np.sum(scores_exp, axis=-1, keepdims=True)
attention = scores_exp / sum_exp

# Weighted sum outputs
# This produces the actual output of attention
out = attention @ V

# ========================================================
# Step 7: Residual connection + LayerNorm
# ========================================================

# ========================================================
# 1) Residual Connection
# ========================================================
#
# The purpose of residual connections is to preserve the
# original token representations while allowing the model
# to incrementally add new, learned information.
#
# Instead of replacing the sequence representation entirely,
# the Transformer refines it:
#     x_out = x_in + f(x_in)
#
# This guarantees stable gradient flow during backpropagation
# and prevents early training stages from destroying useful
# information in the sequence.

x1 = x_seq + out

# ========================================================
# 2) Layer normalization
# ========================================================
#
# Layer Normalization is applied after the residual addition
# to stabilize the scale of activations. It normalizes each
# token independently across the embedding dimension (C),
# ensuring that attention and MLP outputs remain numerically
# well-behaved and trainable.

mean = np.mean(x1, keepdims=True, axis=-1)
var = np.var(x1, keepdims=True, axis=-1)

gamma1 = np.ones(C)
beta1  = np.zeros(C)

center = x1 - mean
normalize = center/np.sqrt(var + 1e-5)
scale = gamma1 * normalize + beta1  


# ========================================================
# Step 8: Feed-forward network (MLP)
# ========================================================

# First going to create our activation function
# Defualting to ReLU is completely fine
def relu(x):
    return np.maximum(0, x)

# So for the forward pass increasing the layers
# SIGNIFICANTLY adds complexity to backprop later which
# is already going to be exceptionally tedious.
# So here 2 linear parts with a activation in between is
# totally fine. 

# Define the hidden layer H_ff
# 4 * C is a standard choice
H_ff = 4 * C

# The input is our token length
input_dim = C

# Set weights and biases with out input and hidden_dim set
# w1 shape(32, 128)
weight1 = np.random.randn(C, H_ff) * np.sqrt(2 / C)
# w1 shape(128, 32) 
# For he intialization here, we need to have np.sqrt(2 / H_ff)
# because this happens after relu
weight2 = np.random.randn(H_ff, C) * np.sqrt(2 / H_ff)

# b1 shape(128,)
bias1 = np.zeros((H_ff,), dtype=np.float32)

# b2 shape(32,)
bias2 = np.zeros((C,), dtype=np.float32)

# Implement the forward pass
Z1 = scale @ weight1 + bias1
A1 = relu(Z1)

# ff_out has shape (12, 8, 32)
ff_out = A1 @ weight2 + bias2


# ========================================================
# Step 9: Residual connection + LayerNorm (Again)
# ========================================================

# Residual connection
x2 = scale + ff_out

# Layer Normalization
# Lots of these variables are re-used from earlier
var2 = np.var(x2, keepdims=True, axis=-1)
mean2 = np.mean(x2, keepdims=True, axis=-1)

gamma2 = np.ones(C)
beta2  = np.zeros(C)

center2 = x2 - mean2
normalize2 = center2/np.sqrt(var2 + 1e-5)
scale2 = gamma2 * normalize2 + beta2


# ========================================================
# Step 10: Output projection → logits (B, T, V)
# ========================================================
#
# After attention, residuals, MLPs, and LayerNorms, scale2
# is the final learned representation of each token.
# Shape: (B, T, C)
#
# These vectors describe *what the model understands* about
# each token, but they are NOT predictions yet.
#
# To predict the next token, we must convert each C-dimensional
# representation into a score for every vocabulary token (V).
#
# This is done with a learned linear projection:
#   (B, T, C) → (B, T, V)
#
# W_vocab learns how internal features map to vocabulary tokens.
# b_vocab provides a bias for each token.
#
# The result logits are raw scores (not probabilities).
# Softmax is applied later during loss computation.
#

W_vocab = np.random.uniform(-0.1, 0.1, size=(C, V))
b_vocab = np.zeros((V,), dtype=np.float32)

logits = scale2 @ W_vocab + b_vocab


# ========================================================
# Step 11: Softmax + Cross-Entropy Loss
# ========================================================
#
# At this point:
#   logits has shape (B, T, V)
#   targets has shape (B, T) and contains integer token IDs
#
# Softmax converts raw logits into probabilities over the vocabulary
# for each token position independently.
#
# probs[b, t, v] = P(next_token = v | context up to position t)

max_logits1 = np.max(logits, axis=-1, keepdims=True)
shifted1 = logits - max_logits1          # numerical stability
exp_scores1 = np.exp(shifted1)
sum_exp1 = np.sum(exp_scores1, axis=-1, keepdims=True)

probs = exp_scores1 / sum_exp1           # (B, T, V)


# --------------------------------------------------------
# Cross-entropy loss
# --------------------------------------------------------
#
# For language modeling, we do NOT compare full distributions.
# Instead, for each (batch, time) position, we:
#   1) Look up the probability assigned to the correct next token
#   2) Take the log of that probability
#   3) Average the negative log-likelihood across all tokens
#
# This measures how confident the model was about the true next token.

b_idx = np.arange(B)                     # batch indices
t_idx = np.arange(T)                     # time indices

# Select the probability of the correct target token at each (b, t)
correct_probs = probs[b_idx[:, None], t_idx[None, :], targets]

# Cross-entropy loss (negative log-likelihood)
loss = -np.mean(np.log(correct_probs + 1e-9))
