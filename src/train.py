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
V_att = x_seq @ Wv



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
out = attention @ V_att

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
# Same pattern as earlier
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
# These vectors describe "what the model understands" about
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


# ========================================================
# 12. Backpropagation (explicit NumPy gradients)
# ========================================================
#
# At this point, I *could* stop the project and still have learned a ton.
# I used to wonder: “Why doesn’t everyone implement Transformer backprop
# from scratch at least once?”
#
# The answer becomes obvious fast: we’re propagating gradients all the way
# from the loss back through every operation to the *actual parameters* —
# including the token embedding table and positional embedding table.
# (Important note: gradients do NOT flow to the integer token IDs themselves;
# they flow to the embedding vectors selected by those IDs.)
#
# Backprop is the process of tracing every mathematical step in reverse to figure
# out which parameters contributed most to the loss. You can think of each parameter
# as a volume knob. Backprop tells us how to turn each knob to reduce the error.
#
# High-level backward structure (reverse of the forward pass):
#   1) Vocab softmax + cross-entropy loss  (loss -> logits gradients)
#   2) Output projection (logits -> final representation)
#   3) LayerNorm2 + Residual2
#   4) Feed-forward network (MLP) backward
#   5) LayerNorm1 + Residual1
#   6) Attention softmax backward (masked softmax)
#   7) Attention scores backward (QK^T / sqrt(H))
#   8) Self-attention backward (out -> attention -> Q,K,V projections)
#   9) Token + positional embedding gradients (scatter-add into tables)
#
# For this section, I’ll probably also post a diagram I draw from scratch to visualize:
#   - backprop flow inside a Transformer block
#   - and self-attention from earlier
#
# The learning loop for this part:
#   1) Understand the structure of the step
#   2) Derive the math for the backward pass
#   3) Implement it in code
#   4) Fail hard the first time
#   5) Debug with help (LLMs + reading)
#   6) Get it correct and verify with sanity checks / gradient checks
#
# This section will be heavily commented to make it easier to follow.
# The difficulty here is less “conceptually hard” and more “tedious and easy to mess up.”
#
# IMPORTANT CONTEXT:
# ------------------------------------------------------------
# This backward pass was implemented as a guided learning exercise.
# Several gradient formulas and structural decisions (especially:
#   - softmax + cross-entropy simplification
#   - LayerNorm backward
#   - tensor flattening / reshaping for linear layers
# ) were derived with external guidance (LLM assistance),
# then carefully reviewed, diagrammed, questioned, and internalized.
#
# The goal here is NOT to claim independent derivation of every formula,
# but to deeply understand:
#   - how gradients flow backward through a Transformer
#   - why shapes must be flattened/merged for matrix math
#   - where gradients split and merge (residual paths)
#   - why this process is tedious even for experts
#
# This section is intentionally verbose and heavily commented
# to serve as a long-term reference and teaching artifact.


# ============================================================
# STEP 11 (BACKWARD): Softmax + Cross-Entropy Loss
# ============================================================
#
# Forward recap:
#   logits: (B, T, V)
#   probs  = softmax(logits)
#   loss   = mean( -log probs[target] )
#
# Key result (well-known in ML):
#   When softmax and cross-entropy are combined, the gradient
#   simplifies to:
#
#     dL/dlogits = (probs - one_hot(targets)) / (B*T)
#
# Interpretation:
#   - Increase probability of the correct token
#   - Decrease probability of all others proportionally
#   - Average gradient across all token positions
#
# This simplification is why frameworks fuse softmax + CE.

# Number of independent prediction sites (loss is averaged)
Npos = B * T

# Start from probabilities
dlogits = probs.copy()                          # (B, T, V)

# Subtract 1 at the correct vocab index for each (b, t)
# This is equivalent to subtracting a one-hot vector,
# but avoids explicitly constructing one.
dlogits[b_idx[:, None], t_idx[None, :], targets] -= 1.0

# Account for mean reduction in the loss
dlogits /= Npos


# ============================================================
# STEP 10 (BACKWARD): Output Projection (Linear Layer)
# ============================================================
#
# Forward:
#   logits = scale2 @ W_vocab + b_vocab
#
# Shapes:
#   scale2 : (B, T, C)
#   W_vocab: (C, V)
#   logits : (B, T, V)
#
# Backward responsibilities:
#   1) Accumulate bias gradients (sum over B and T)
#   2) Compute weight gradients (requires flattening B,T)
#   3) Propagate gradient back into scale2

# Bias affects every (b,t) equally → sum over batch + time
db_vocab = np.sum(dlogits, axis=(0, 1))         # (V,)

# Flatten (B,T) so each token position is treated as
# an independent training example for the linear layer
N = B * T
X = scale2.reshape(N, C)                        # (BT, C)
G = dlogits.reshape(N, V)                       # (BT, V)

# Weight gradient follows standard linear layer rule
dW_vocab = X.T @ G                              # (C, V)

# Gradient flowing back into transformer representation
dscale2 = dlogits @ W_vocab.T                  # (B, T, C)


# ============================================================
# STEP 9 (BACKWARD): LayerNorm2 + Residual Connection
# ============================================================
#
# Forward:
#   x2      = scale + ff_out
#   scale2  = LayerNorm(x2)
#
# Backward tasks:
#   1) Compute gamma/beta gradients
#   2) Backprop through normalization constraints
#   3) Split gradient across residual branches
#
# LayerNorm is difficult because it enforces:
#   - zero mean
#   - unit variance
# per token.
#
# Backward must undo those constraints, which is why
# the formula looks dense.

eps = 1e-5
Cdim = C

# Gamma and beta gradients:
#   beta shifts → sum of gradients
#   gamma scales normalized input
dbeta2  = np.sum(dscale2, axis=(0,1))                 # (C,)
dgamma2 = np.sum(dscale2 * normalize2, axis=(0,1))    # (C,)

# Cached forward quantities
xhat = normalize2                                     # (B, T, C)
std_inv = 1.0 / np.sqrt(var2 + eps)                   # (B, T, 1)

# Gradient through affine LN step
dxhat = dscale2 * gamma2                              # (B, T, C)

# Core LayerNorm backward formula
dx2 = (1.0 / Cdim) * std_inv * (
        Cdim * dxhat
        - np.sum(dxhat, axis=-1, keepdims=True)
        - xhat * np.sum(dxhat * xhat, axis=-1, keepdims=True)
     )

# Residual split:
# x2 = scale + ff_out
dscale  = dx2
dff_out = dx2


# ============================================================
# STEP 8 (BACKWARD): Feed-Forward Network (MLP)
# ============================================================
#
# Forward:
#   Z1     = scale @ weight1 + bias1
#   A1     = ReLU(Z1)
#   ff_out = A1 @ weight2 + bias2
#
# This section is mechanically similar to a standard NN,
# but complicated by tensor shapes (B, T, ·).
#
# The key difficulty is flattening (B,T) so matrix
# multiplications are mathematically valid, then reshaping back.

BT = B * T

# ---- Backprop through second linear layer ----

# Bias gradient: sum over all token positions
db2 = np.sum(dff_out, axis=(0,1))                     # (C,)

# Flatten tokens to compute weight gradient
A_flat = A1.reshape(BT, H_ff)                         # (BT, H_ff)
G_flat = dff_out.reshape(BT, C)                       # (BT, C)
dW2 = A_flat.T @ G_flat                               # (H_ff, C)

# Gradient flowing back into hidden activations
dA1 = dff_out @ weight2.T                             # (B, T, H_ff)

# ---- ReLU backward ----
# Gradient only flows where activation was positive
dZ1 = dA1 * (Z1 > 0)                                  # (B, T, H_ff)

# ---- Backprop through first linear layer ----

# Bias gradient
db1 = np.sum(dZ1, axis=(0,1))                         # (H_ff,)

# Weight gradient (flatten again)
S_flat  = scale.reshape(BT, C)                        # (BT, C)
G1_flat = dZ1.reshape(BT, H_ff)                       # (BT, H_ff)
dW1 = S_flat.T @ G1_flat                              # (C, H_ff)

# Gradient flowing back into 'scale' from MLP branch
dscale_mlp = dZ1 @ weight1.T                          # (B, T, C)

# Merge gradients at residual junction
# scale participates in BOTH:
#   - residual path
#   - MLP path
dscale = dscale + dscale_mlp


# ============================================================
# STEP 7 (BACKWARD): LayerNorm1 + Residual1
# ============================================================
#
# Forward:
#   x1    = x_seq + out
#   scale = LayerNorm(x1)   # (this "scale" feeds the MLP)
#
# At this point we have:
#   dscale  : gradient flowing into LN1 output (B, T, C)
#
# We need:
#   - gradients for LN1 parameters: gamma1, beta1
#   - gradient flowing back into x1: dx1
#   - split dx1 into residual branches:
#       dx_seq += dx1
#       dout   += dx1
#
# NOTE:
# LN backward looks "dense" because it enforces per-token constraints:
#   - subtract mean
#   - divide by std
# which introduces coupling across the feature dimension C.

eps = 1e-5
Cdim = C

# gamma1/beta1 gradients
dbeta1  = np.sum(dscale, axis=(0, 1))                # (C,)
dgamma1 = np.sum(dscale * normalize, axis=(0, 1))    # (C,)

# Backprop through LN1 affine
dxhat1 = dscale * gamma1                              # (B, T, C)

# Cached forward quantities from LN1
xhat1 = normalize                                     # (B, T, C)
std_inv1 = 1.0 / np.sqrt(var + eps)                   # (B, T, 1)

# LayerNorm backward formula (same as LN2, different cached vars)
dx1 = (1.0 / Cdim) * std_inv1 * (
        Cdim * dxhat1
        - np.sum(dxhat1, axis=-1, keepdims=True)
        - xhat1 * np.sum(dxhat1 * xhat1, axis=-1, keepdims=True)
     )                                                # (B, T, C)

# (Optional sanity check: should be ~0)
print("max|sum_c dx1|:", np.max(np.abs(dx1.sum(axis=-1))))

# Residual split for x1 = x_seq + out
dx_seq = dx1          # gradient into x_seq from residual
dout   = dx1          # gradient into attention output "out"


# ============================================================
# STEP 6 (BACKWARD): Self-Attention
# ============================================================
#
# Forward recap:
#   Q = x_seq @ Wq                 (B, T, H)
#   K = x_seq @ Wk                 (B, T, H)
#   V_values = x_seq @ Wv          (B, T, H)
#
#   scores = (Q @ K.T) / sqrt(H)   (B, T, T)
#   scores_masked = apply causal mask (set future to -1e9)
#   attention = softmax(scores_masked)   (B, T, T)
#   out = attention @ V_values           (B, T, H)
#
# Backward tasks:
#   1) out = attention @ V_values
#   2) attention = softmax(scores_masked)
#   3) scores = (Q @ K.T) / sqrt(H)
#   4) Q,K,V projections back to x_seq and weights Wq,Wk,Wv
#
# The "hard" part is bookkeeping:
#   - transposes
#   - masking
#   - shape-consistent matmuls

# ------------------------------------------------------------
# 6.1 Backprop through out = attention @ V_values
# ------------------------------------------------------------
#
# out[b] = attention[b] @ V_values[b]
# So:
#   dattention[b] = dout[b] @ V_values[b].T
#   dV_values[b]  = attention[b].T @ dout[b]

V_T = np.transpose(V_values, (0, 2, 1))                 # (B, H, T)
dattention = dout @ V_T                                  # (B, T, T)

att_T = np.transpose(attention, (0, 2, 1))              # (B, T, T)
dV_values = att_T @ dout                                 # (B, T, H)

# ------------------------------------------------------------
# 6.2 Backprop through attention = softmax(scores_masked)
# ------------------------------------------------------------
#
# Softmax backward (row-wise):
# If a = softmax(s), then:
#   ds = a * (da - sum(da * a))
#
# IMPORTANT:
# scores_masked had -1e9 inserted for future positions.
# Those masked entries are constants, so their gradients must be 0.

# softmax backward for each row
dscores_masked = attention * (
    dattention - np.sum(dattention * attention, axis=-1, keepdims=True)
)                                                       # (B, T, T)

# zero out gradients for masked positions (future tokens)
dscores_masked[:, causal_mask] = 0.0

# ------------------------------------------------------------
# 6.3 Backprop through scaling: scores = raw_scores / sqrt(H)
# ------------------------------------------------------------
dscores = dscores_masked / np.sqrt(H)                   # (B, T, T)

# ------------------------------------------------------------
# 6.4 Backprop through raw_scores = Q @ K.T
# ------------------------------------------------------------
#
# raw_scores[b] = Q[b] @ K[b].T
# So:
#   dQ[b] = dscores[b] @ K[b]
#   dK[b] = dscores[b].T @ Q[b]

dQ = dscores @ K                                         # (B, T, H)
dscores_T = np.transpose(dscores, (0, 2, 1))             # (B, T, T)
dK = dscores_T @ Q                                       # (B, T, H)

# ------------------------------------------------------------
# 6.5 Backprop through projections:
#   Q = x_seq @ Wq
#   K = x_seq @ Wk
#   V_values = x_seq @ Wv
# ------------------------------------------------------------
#
# Flattening trick again:
#   treat each (b,t) token position as one sample.

BT = B * T
X_flat = x_seq.reshape(BT, C)                            # (BT, C)

dQ_flat = dQ.reshape(BT, H)                               # (BT, H)
dK_flat = dK.reshape(BT, H)                               # (BT, H)
dV_flat = dV_values.reshape(BT, H)                        # (BT, H)

# Weight gradients
dWq = X_flat.T @ dQ_flat                                  # (C, H)
dWk = X_flat.T @ dK_flat                                  # (C, H)
dWv = X_flat.T @ dV_flat                                  # (C, H)

# Gradient back into x_seq from each projection path
dx_from_Q = dQ @ Wq.T                                     # (B, T, C)
dx_from_K = dK @ Wk.T                                     # (B, T, C)
dx_from_V = dV_values @ Wv.T                              # (B, T, C)

# Total gradient into x_seq combines:
#   - residual path (dx_seq from Step 7)
#   - projection paths (Q/K/V)
dx_seq = dx_seq + dx_from_Q + dx_from_K + dx_from_V        # (B, T, C)


# ============================================================
# STEP 5 (BACKWARD): Token Embeddings + Positional Embeddings
# ============================================================
#
# Forward:
#   tok = token_embedding_table[inputs]      # (B, T, C)
#   pos = positional_embedding_table[positions]  # (T, C)
#   x_seq = tok + pos                        # (B, T, C)
#
# Backward:
#   dtok = dx_seq
#   dpos = sum over batch (because pos is broadcast across B)
#   Then scatter-add dtok back into the embedding tables.

# Gradient into tok and pos from x_seq = tok + pos
dtok = dx_seq                                             # (B, T, C)
dpos = np.sum(dx_seq, axis=0)                              # (T, C)

# ------------------------------------------------------------
# Backprop into positional_embedding_table
# ------------------------------------------------------------
dpositional_embedding_table = np.zeros_like(positional_embedding_table)  # (T, C)
dpositional_embedding_table[positions] += dpos

# ------------------------------------------------------------
# Backprop into token_embedding_table via scatter-add
# ------------------------------------------------------------
#
# token_embedding_table is indexed by integer token IDs.
# Gradients do not flow into integer IDs, only into the rows
# of the embedding table that were selected.
#
# This is a classic "scatter-add" situation.

dtoken_embedding_table = np.zeros_like(token_embedding_table)  # (V, C)

# Efficient scatter-add: add dtok[b,t] into row token_id = inputs[b,t]
np.add.at(dtoken_embedding_table, inputs, dtok)
