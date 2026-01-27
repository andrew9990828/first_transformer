# Transformer From Scratch (Pure NumPy)

This repo contains a minimal, single-head Transformer language model built entirely from scratch using NumPy.

**No PyTorch. No autograd.**  
Every forward pass, backward pass, and parameter update is written explicitly.

The goal wasn't to build something impressive — it was to build something I could eventually explain end-to-end without hiding behind a framework.

## Why I Built This

I kept hearing explanations of Transformers that felt hand-wavy.

So I wanted to answer a simpler question:

> *If I remove all abstractions, can I still make a Transformer learn — and understand why it works?*

The only way I knew how to do that was to write everything manually and let it break until it didn't.

## Important Context (Read This)

**I cannot yet explain and defend every single line of this code from memory.**

That is intentional.

This repo marks the end of the *build phase*, not the end of understanding.  
The next step is a full review pass where I revisit each section, redraw diagrams, and close the remaining gaps.

**This project exists so that review is possible.**

## What This Model Is

- Character-level autoregressive language model
- Single Transformer block
- Single self-attention head
- Residual connections + LayerNorm
- Feed-forward MLP
- Softmax cross-entropy loss
- Trained with plain SGD

**Input:** last T characters  
**Output:** probability distribution over the next character

## Results

After 100 epochs with 10,000 steps per epoch, this was the final 200-token generation:
```
Once upon thee
                Quoth the chamber door,
That this soul with many a quaint and stern decorum said I, "thing more.

Then that thee—by that melancholy burden bore;
    But these air grew denser, p
```

This is produced by a tiny model with:
- an 8-character context window
- a single attention head
- no pretrained weights
- no ML frameworks

The model clearly learns:
- punctuation and spacing
- line breaks and indentation
- recurring Poe-specific phrases
- stylistic cadence

## How It Works (High Level)

1. Characters are mapped to embeddings
2. Positional embeddings are added
3. Self-attention lets each token weight earlier tokens
4. Residual + LayerNorm stabilize learning
5. A small MLP refines representations
6. Output logits predict the next character
7. Cross-entropy loss measures error
8. Gradients propagate backward through everything

Nothing fancy — just careful math and a lot of bookkeeping.

## How to Run
```bash
python src/train.py
```

**Requirements:**
- Python 3.7+
- NumPy

That's it. No other dependencies.

## Backpropagation & LLM Assistance (Transparency)

I want to be explicit about this.

The forward pass and overall architecture were implemented independently.

The backward pass was implemented with LLM assistance, especially for:
- LayerNorm backward
- Softmax + cross-entropy simplifications
- Attention gradient bookkeeping

**This wasn't copy-paste.**

Each backward section was reviewed, debugged, reshaped, and sanity-checked until training behaved correctly. The goal was not independent derivation — it was understanding gradient flow well enough to reason about it.

That understanding is still being solidified during review.

## What This Taught Me (So Far)

A few things became very clear:

- Loss can drop long before outputs look readable
- Character-level models learn local statistics first
- Self-attention isn't mystical — it's scaled dot products plus softmax
- Residual connections aren't optional
- LayerNorm backward is genuinely hard
- Automatic differentiation is a gift

**Most importantly:**  
*Transformers aren't magic — they're disciplined signal routing.*

## What This Is Not

This is not production-ready and isn't meant to be.

- Single attention head
- Tiny context window
- Character-level tokens
- No dropout, warmup, or advanced optimizers

Those constraints are deliberate. They force understanding instead of scale.

## Files
```
.
├── src/
│   └── train.py                      # Full implementation
├── data_textfiles/
│   └── the_raven_edgarallanpoe.txt  # Training corpus
└── README.md                         # This file
```

## Next Steps

- Line-by-line review of the entire model
- Hand-drawn diagrams for:
  - Transformer block
  - Self-attention
  - Backpropagation paths
- A separate write-up documenting what failed early and why

**If I can explain this on a whiteboard without notes, the project did what it was supposed to do.**