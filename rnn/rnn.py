from math import tanh
import numpy as np

def init_rnn_forward_params(vocab_size, dim, max_len, seed=42): 
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.01, (vocab_size, dim))
    W_x = rng.normal(0, 0.01, (dim, dim))
    W_h = rng.normal(0, 0.01, (dim, dim))
    P = rng.normal(0, 0.01, (max_len, dim)) # learned positions for now

    return E, W_x, W_h, P


def rnn_forward(E, W_x, W_h, P, token_ids): 
    dim = W_h.shape[0]
    h_prev = np.zeros(dim)

    hs = [] # Store hidden states
    xs = [] # store inputs (for later backpropogation)

    for i, token_id in enumerate(token_ids): 
        x = E[token_id] + P[i]
        h = np.tanh(W_h @ h_prev + W_x @ x)

        xs.append(x)
        hs.append(h)

        h_prev = h
    return xs, hs


if __name__ == "__main__":
    vocab_size = 10
    dim = 8
    token_ids = [1, 3, 5, 2, 4, 3]
    max_len = len(token_ids)

    E, W_x, W_h, P = init_rnn_forward_params(vocab_size, dim, max_len)

    xs, hs = rnn_forward(E, W_x, W_h, P, token_ids)

    print("Number of steps:", len(hs))
    print("Hidden state shape:", hs[0].shape)
    print("Last hidden state:", hs[-1])