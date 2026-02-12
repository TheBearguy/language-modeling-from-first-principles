#!/usr/bin/env python3
"""
Entry point for training skip-gram embeddings.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe import train, encode
from skipgram import train_skipgram


def main():
    # Sample text for BPE training
    text = "The penguin started heading towards the mountains; some 70 kms away"
    text = text.lower()
    
    # Train BPE
    vocab, merge_rules, vocab_to_id, corpus = train(text, target=15)
    
    # Encode the text
    token_ids, _ = encode(text, merge_rules, vocab_to_id)
    
    # Train skip-gram
    E = train_skipgram(
        token_ids, 
        vocab_size=len(vocab_to_id), 
        dim=32
    )
    
    print(f"\nSkip-gram training complete. Embedding matrix shape: {E.shape}")


if __name__ == "__main__":
    main()
