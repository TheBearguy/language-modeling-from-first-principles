#!/usr/bin/env python3
"""
Entry point for training position-aware language model.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe import train, encode
from positional_encoding import train_language_model_with_positions


def main():
    # -------- 1. Raw training text --------
    text = (
        "the cat sat on the mat "
        "the dog sat on the log "
        "the cat chased the dog "
    )

    # -------- 2. Train BPE --------
    vocab, merge_rules, vocab_to_id, _ = train(text, target=50)

    # -------- 3. Encode text --------
    token_ids, _ = encode(text, merge_rules, vocab_to_id)

    print("Token IDs:", token_ids)
    print("Vocab size:", len(vocab_to_id))

    # -------- 4. Train LM with LEARNED positions --------
    print("\nTraining with LEARNED positional embeddings\n")

    E_learned, W_learned, P_learned = train_language_model_with_positions(
        token_ids=token_ids,
        vocab_size=len(vocab_to_id),
        dim=32,
        max_len=len(token_ids),
        epochs=20,
        lr=0.05,
        positional_mode="learned"
    )

    # -------- 5. Train LM with SINUSOIDAL positions --------
    print("\nTraining with SINUSOIDAL positional embeddings\n")

    E_sin, W_sin, P_sin = train_language_model_with_positions(
        token_ids=token_ids,
        vocab_size=len(vocab_to_id),
        dim=32,
        max_len=len(token_ids),
        epochs=20,
        lr=0.05,
        positional_mode="sinusoidal"
    )


if __name__ == "__main__":
    main()
