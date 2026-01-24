# Language Models From First Principles· Embeddings

A **from-scratch educational implementation** of core NLP building blocks: Byte Pair Encoding (BPE), skip-gram embeddings, simple language models, positional encoding, and recurrent language models. No PyTorch/TensorFlow—just NumPy and first-principles math.

---

## Overview

The repo implements the pipeline outlined in `bpe/start.md`:

1. **Phase 1 — Tokenization:** Build a BPE tokenizer from scratch (`train` → `encode` → `decode`).
2. **Phase 2 — Embeddings:** Train token embeddings via **skip-gram** (Word2Vec-style) or via **language modeling** (next-token prediction).

Each module includes both code and markdown docs that derive the math and mechanics step by step.

---

## Repository Structure

```
bpe/
├── bpe/                    # BPE tokenizer
│   ├── bpe.py              # train, encode, decode
│   └── start.md            # Phase 1 & 2 roadmap
├── skipgram/               # Skip-gram embeddings (Word2Vec-style)
│   ├── skip_gram.py        # negative sampling, (E, O) matrices
│   └── skip-gram.md        # concept & gradient derivations
├── language_model/         # Simple neural LM (GPT-style logic)
│   ├── language_model.py   # bag-of-words average → softmax
│   └── language_modelling.md
├── positional_encoding/    # Position-aware LM
│   ├── lm_position_encoding.py   # learned vs sinusoidal positions
│   └── positional_encoding.md
├── rnn/                    # RNN language model
│   ├── __init__.py
│   ├── rnn.py              # tanh RNN, BPTT, optional positions
│   └── rnn_concept.md
└── README.md
```

---

## Modules

### 1. `bpe` — Byte Pair Encoding

- **`train(text, target)`** — Learn merge rules from text up to `target` vocab size.
- **`encode(text, merge_rules, vocab_to_id)`** — Tokenize text → token IDs (and token list).
- **`decode(token_ids, vocab_to_id)`** — Map token IDs back to text.

Tokenization: words + punctuation; words split into characters with `</w>`; punctuation kept atomic. Supports `<UNK>` for unknown tokens.

**Run:** `python -m bpe.bpe` (trains on sample text, encodes/decodes).

---

### 2. `skipgram` — Skip-gram Embeddings

- **`build_skipgram_pairs(token_ids, window_size)`** — (center, context) pairs from a token sequence.
- **`train_skipgram(...)`** — Train input matrix **E** and output matrix **O** with **negative sampling** (sigmoid + binary loss).

Embeds “meaning” via **distributional similarity**: tokens in similar contexts get similar vectors. No sequence order; local, symmetric context.

**Run:** `python -m skipgram.skip_gram` (uses BPE `encode` output; requires `bpe`).

---

### 3. `language_model` — Simple Neural LM

- **`init_lm_params(vocab_size, dim)`** — Embedding matrix **E**, output projection **W**.
- **`lm_steps(E, W, prefix_ids, target_id, lr)`** — One step: average prefix embeddings → logits → softmax → cross-entropy loss; backward updates **E** and **W**.
- **`train_language_model(token_ids, vocab_size, dim, lr, epochs)`** — Next-token prediction loop.

Uses **bag-of-words averaging** over the prefix (no positional encoding). Conceptually GPT-like (predict next token), but simpler.

**Run:** `python -m language_model.language_model` (sanity check on fake token IDs).

---

### 4. `positional_encoding` — Position-Aware LM

- **`init_learned_positions(max_len, dim)`** — Random **P**, updated by gradients.
- **`init_sinusoidal_positions(max_len, dim)`** — Fixed sin/cos scheme (no learning).
- **`lm_with_positions(E, W, P, ...)`** — Forward: `x_k = E[token] + P[k]`, then average → logits → softmax.
- **`train_language_model_with_positions(..., positional_mode="learned"|"sinusoidal")`** — Full training with positions.

Solves the “**order blindness**” of mean-pooling: same tokens in different order now produce different representations. See `positional_encoding.md` for learned vs sinusoidal trade-offs.

**Run:** `python -m positional_encoding.lm_position_encoding` (BPE on sample text → encode → train LM with learned then sinusoidal positions).

---

### 5. `rnn` — RNN Language Model

- **`init_rnn_params(vocab_size, dim)`** — **E**, **W_x**, **W_h**, **W_o**.
- **`init_learned_positions(max_len, dim)`** / **`init_sinusoidal_positions(max_len, dim)`** — Position embeddings **P**.
- **`rnn_lm_forward(E, W_x, W_h, W_o, P, token_ids)`** — `x_k = E[token] + P[k]`, then `h_i = tanh(W_h h_{i-1} + W_x x_i)`, `logits = h_i @ W_o`.
- **`rnn_lm_backward(...)`** — Backprop-through-time; updates **E**, **W_x**, **W_h**, **W_o**, and optionally **P**.
- **`train_rnn_language_model(...)`** — Full training loop with `positional_mode="learned"|"sinusoidal"`.

Imports **BPE** (`train`, `encode`) for the default pipeline. The `__main__` block runs a full demo and sanity-checks next-token prediction. Replaces mean-pooling with a **recurrent hidden state**: order is inherent, recency bias emerges from the recurrence. See `rnn_concept.md` for vanishing gradients, fixed memory size, and why attention eventually supersedes RNNs.

**Run:** `python -m rnn.rnn` — BPE on sample text → encode → train RNN LM with learned positions → sanity-check next-token prediction.

---

## Dependencies

- **Python 3**
- **NumPy**

```bash
pip install numpy
```

---

## Quick Start

**1. BPE only**

```bash
python -m bpe.bpe
```

**2. BPE → LM with positional encoding**

```bash
python -m positional_encoding.lm_position_encoding
```

**3. Skip-gram** — Operates on BPE-encoded `token_ids`. The `skip_gram` module imports from `bpe`; wire up BPE `train`/`encode` and pass `token_ids` + `vocab_size` into `train_skipgram` (see `skip_gram.py`).

**4. RNN LM** — Full demo: BPE → encode → train RNN LM (learned positions) → predict next token.

```bash
python -m rnn.rnn
```

Or use `train_rnn_language_model` from `rnn.rnn` with your own `token_ids`, `vocab_size`, and `max_len`.

---

## Data Flow (Conceptual)

```
text → [BPE] → tokens / token_ids
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
  [Skip-gram]            [Language model]
  (E, O)                  (E, W; optional P, RNN)
        ↓                       ↓
  embeddings              next-token prediction
```

---

## Design Notes

- **BPE:** Uses Python character boundaries. Docs note open questions: emojis, accents, mixed scripts (e.g. Hindi)—production tokenizers (e.g. GPT-style) handle Unicode more carefully.
- **LM vs skip-gram:** LM = causal, ordered, “what comes next?”; skip-gram = local, unordered, “what appears near me?”. Different learning signals, different use cases.
- **Positions:** Learned **P** = flexible, task-specific; sinusoidal **P** = fixed, better length extrapolation, fewer parameters. See `positional_encoding.md` for the bias–variance trade-off.

---

## References in Repo

- `bpe/start.md` — Phases 1 & 2, high-level pipeline.
- `language_model/language_modelling.md` — LM math, backward pass, comparison to skip-gram.
- `positional_encoding/positional_encoding.md` — Why position breaks symmetry; learned vs sinusoidal.
- `rnn/rnn_concept.md` — RNN mechanics, limitations, LSTM/attention context.
- `skipgram/skip-gram.md` — Skip-gram mechanics, negative sampling, E vs O.

---

## License

See repository license, if present.
