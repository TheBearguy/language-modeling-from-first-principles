from math import tanh
from bpe.bpe import train, encode
import numpy as np
from language_model.language_model import softmax

def init_rnn_params(vocab_size, dim, seed=42):
    rng = np.random.default_rng(seed)

    E = rng.normal(0, 0.01, size=(vocab_size, dim))
    W_x = rng.normal(0, 0.01, size=(dim, dim))
    W_h = rng.normal(0, 0.01, size=(dim, dim))
    W_o = rng.normal(0, 0.01, size=(dim, vocab_size))

    return E, W_x, W_h, W_o


def init_learned_positions(max_len, dim, seed=42):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.01, size=(max_len, dim))
    return P


def init_sinusoidal_positions(max_len, dim):
    P = np.zeros((max_len, dim))

    for k in range(max_len):
        for j in range(0, dim, 2):
            angle = k / (10000 ** (j / dim))
            P[k, j] = np.sin(angle)
            if j + 1 < dim:
                P[k, j + 1] = np.cos(angle)

    return P


def rnn_lm_forward(E, W_x, W_h, W_o, token_ids): 
    dim = W_h.shape[0]
    h_prev = np.zeros(dim)

    hs = [] # Store hidden states
    xs = [] # store inputs (for later backpropogation)
    ps = [] 

    for i, token_id in enumerate(token_ids): 
        x = E[token_id] + P[i]
        h = np.tanh(W_h @ h_prev + W_x @ x)

        logits = h @ W_o
        p = softmax(logits)

        xs.append(x)
        hs.append(h)
        ps.append(p)

        h_prev = h
    return xs, hs
    

def rnn_lm_backward(
    E, W_x, W_h, W_o, P, 
    xs, hs, ps, token_ids, 
    lr, learn_positions = True
): 
    dim = W_h.shape[0]
    dh_next = np.zeros(dim)

    for i in reversed(range(len(hs))): 
        target = token_ids[i+1]

        # Gradient with respect to logic
        dp = ps[i].copy()
        dp[target] = dp[target] - 1 # p - y

        # Output layer 
        W_o = W_o - lr * np.outer(hs[i], dp)

        # Backprop into h
        dh = W_o @ dp + dh_next

        # Backprop through tanh
        dtanh = dh * (1 - hs[i] ** 2)

        # Update recurrent weights: 
        W_x = W_x - lr * np.outer(dtanh, xs[i])
        W_h = W_h - lr * np.outer(dtanh, hs[i-1] if i>0 else 0)

        # Update embeddings
        E[token_ids[i]] = E[token_ids[i]] - lr * dtanh

        if learn_positions: 
            P[i] = P[i] - lr * dtanh

        # Propogate to previus step
        dh_next = W_h.T @ dtanh


def train_rnn_language_model(
    token_ids,
    vocab_size,
    dim=32,
    max_len=128,
    lr=0.05,
    epochs=10,
    positional_mode="learned"
): 
    E, W_x, W_h, W_o = init_rnn_params(vocab_size, dim)
    if positional_mode == "learned":
            P = init_learned_positions(max_len, dim)
            learn_positions = True
    else:
        P = init_sinusoidal_positions(max_len, dim)
        learn_positions = False

    for epoch in range(epochs):
        xs, hs, ps = rnn_lm_forward(E, W_x, W_h, W_o, P, token_ids)

        rnn_lm_backward(
            E, W_x, W_h, W_o, P,
            xs, hs, ps, token_ids,
            lr, learn_positions
        )

        loss = sum(-np.log(ps[i][token_ids[i + 1]]) for i in range(len(ps)))
        print(f"epoch {epoch+1}, loss {loss:.4f}")

    return E, W_x, W_h, W_o, P 


if __name__ == "__main__":
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