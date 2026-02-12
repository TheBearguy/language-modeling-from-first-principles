#!/usr/bin/env python3
"""
Entry point for training RNN language model.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bpe import train, encode
from rnn import train_rnn_language_model
from language_model import softmax


def main():
    text = (
        "the cat sat on the mat "
        "the dog sat on the log "
        "the cat chased the dog "
    )

    # 2. Train BPE (once)
    vocab, merge_rules, vocab_to_id, _ = train(text, target=50)
    token_ids, _ = encode(text, merge_rules, vocab_to_id)

    print("Vocab size:", len(vocab_to_id))
    print("Token IDs:", token_ids)

    # 3. RNN LM hyperparameters
    dim = 32
    lr = 0.05
    epochs = 20
    max_len = len(token_ids)

    # 4. Train RNN LM (LEARNED positions)
    print("\nTraining RNN LM with LEARNED positional embeddings\n")

    E, W_x, W_h, W_o, P = train_rnn_language_model(
        token_ids=token_ids,
        vocab_size=len(vocab_to_id),
        dim=dim,
        max_len=max_len,
        lr=lr,
        epochs=epochs,
        positional_mode="learned"
    )

    # 5. sanity check: predict next token
    def predict_next_rnn(E, W_x, W_h, W_o, P, prefix_ids):
        h = np.zeros(W_h.shape[0])

        for i, t in enumerate(prefix_ids):
            x = E[t] + P[i]
            h = np.tanh(W_h @ h + W_x @ x)

        probs = softmax(h @ W_o)
        return np.argmax(probs)

    id_to_vocab = {i: t for t, i in vocab_to_id.items()}

    prefix = token_ids[:5]
    pred_id = predict_next_rnn(E, W_x, W_h, W_o, P, prefix)

    print("\nPrefix token IDs:", prefix)
    print("Predicted next token:", id_to_vocab[pred_id])


if __name__ == "__main__":
    main()
