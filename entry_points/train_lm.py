#!/usr/bin/env python3
"""
Entry point for training simple language model.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language_model import train_language_model


def main():
    # Sample text for BPE training
    text = (
        "the cat sat on the mat "
        "the dog sat on the log "
        "the cat chased the dog "
    )
    
    # Import here to avoid circular import issues
    from bpe import train, encode
    
    # Train BPE
    vocab, merge_rules, vocab_to_id, _ = train(text, target=50)
    
    # Encode text
    token_ids, _ = encode(text, merge_rules, vocab_to_id)
    
    print("Token IDs:", token_ids)
    print("Vocab size:", len(vocab_to_id))
    
    # Train language model
    E, W = train_language_model(
        token_ids=token_ids,
        vocab_size=len(vocab_to_id),
        dim=32,
        lr=0.05,
        epochs=20
    )
    
    print(f"\nTraining complete.")
    print("Embedding matrix shape:", E.shape)
    print("Output matrix shape:", W.shape)


if __name__ == "__main__":
    main()
