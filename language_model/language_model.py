import numpy as np

# from bpe.bpe import vocab_to_id, text, merge_rules


def init_lm_params(vocab_size, dim, seed = 42): 
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.01, (vocab_size, dim)) # token embeddigns
    W = rng.normal(0, 0.01, (dim, vocab_size)) # output projections
    return E, W


def softmax(x): 
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)


def lm_steps(E, W, prefix_ids, target_id, lr): 
    # Forward: 
    h = E[prefix_ids].mean(axis=0)
    logits = np.dot(h, W)
    probs = softmax(logits)

    loss = - np.log(probs[target_id])
    # Backward: 
    dlogits = probs
    dlogits[target_id] -= 1

    # gradient
    dw = np.outer(h, dlogits)
    dh = np.dot(W, dlogits)

    # Update output weights
    W = W - lr*dw

    # distribute gradient evenly to prefix embeddings
    gradient_per_token = dh/len(prefix_ids)
    for t in prefix_ids: 
        E[t] = E[t] - lr * gradient_per_token
    return loss

def train_language_model(
    token_ids,
    vocab_size,
    dim=32,
    lr=0.05,
    epochs=10
):
    E, W = init_lm_params(vocab_size, dim)

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(1, len(token_ids)):
            prefix = token_ids[:i]
            target = token_ids[i]
            loss = lm_steps(E, W, prefix, target, lr)
            total_loss += loss

        print(f"epoch {epoch+1}, loss {total_loss:.4f}")

    return E, W


# if __name__ == "__main__": 
#     token_ids, _ = (text, merge_rules, vocab_to_id)

# E, W = train_language_model(
#     token_ids,
#     vocab_size=len(vocab_to_id),
#     dim=32,
#     epochs=20
# )

