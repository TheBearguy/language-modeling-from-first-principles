#!/usr/bin/env python3
"""
Entry point for training and testing BPE tokenizer.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe import train, encode, decode


def main():
    target = 15
    # text = "mountain"
        # "mountains"
        # "mountainous"
        # "mountain-like"
        # "Mountain"
        # "MOUNTAIN"
    
    text = "The penguin started heading towards the mountains; some 70 kms away"
    # Train once: 
    text = text.lower()
    vocab, merge_rules, vocab_to_id, corpus = train(text, target)

    print(f"\nEncoding NEW TEXT\n")

    # Now encode new text using the learned rules: 
    new_text = "penguin heading towards the mountains, for a purpose"
    # new_text = "mountain-like"
    token_ids, encoded_corpus = encode(new_text, merge_rules, vocab_to_id)
    print(f"\nNew text: {new_text}\n")
    print(f"\nEncoded tokens: {encoded_corpus}\n")
    print(f"\nToken IDs: {token_ids}\n")
    print(f"\n Decoding token id to string\n")
    text = decode(token_ids, vocab_to_id)
    print(text)


if __name__ == "__main__":
    main()
