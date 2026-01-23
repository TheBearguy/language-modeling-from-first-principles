import numpy as np
from bpe.bpe import encode, train
from language_model.language_model import init_lm_params, softmax

def init_learned_positions(max_len, dim, seed=42): 
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.01, (max_len, dim))
    # This will be updated by graidents
    return P


def init_sinusoidal_positions(max_len, dim): 
    P = np.zeros((max_len, dim))
    for k in range(max_len): 
        for j in range(0, dim, 2): 
            angle = k / (10000 ** (j/dim))
            P[k,j] = np.sin(angle)
            if j + 1 < dim: 
                P[k,j+1] = np.cos(angle)
    return P


def lm_with_positions(
    E, W, P, 
    prefix_ids, target_id,
    lr, 
    learned_positions = True
): 
    # Forward propogation
    vectors = []
    for idx, token in enumerate(prefix_ids): 
        vectors.append(E[token] + P[idx])
    
    H = np.mean(vectors, axis=0)
    logits = H @ W
    probs = softmax(logits)
    loss = -np.log(probs[target_id])

    # Backward prop
    dlogits = probs
    dlogits[target_id] = dlogits[target_id] - 1

    dW = np.outer(H, dlogits)
    dH = W @ dlogits

    W = W - lr * dW

    gradient = dH/len(prefix_ids)

    for idx, token in enumerate(prefix_ids): 
        E[token] = E[token] - lr * gradient
        if learned_positions: 
            P[idx] = P[idx] - lr * gradient
    return loss


def train_language_model_with_positions(
    token_ids, 
    vocab_size, 
    dim = 32, 
    max_len = 128, 
    lr = 0.05, 
    epochs = 10, 
    positional_mode = "learned"
): 
    E, W = init_lm_params(vocab_size, dim)
    if positional_mode == "learned": 
        P = init_learned_positions(max_len, dim) 
        learn_positions = True
    elif positional_mode == "sinusoidal": 
        P = init_sinusoidal_positions(max_len, dim)
        learn_positions = False
    else: 
        raise ValueError("invalid positonal_mode")
    
    for epoch in range(epochs): 
        total_loss = 0.0
        for i in range(1, len(token_ids)): 
            prefix = token_ids[:i]
            target = token_ids[i]
            loss = lm_with_positions(
                E, W, P, 
                prefix, target, 
                lr, 
                learn_positions
            )
            total_loss += loss
        print(f"epoch {epoch+1}, loss {total_loss:.4f}")
    return E, W, P


if __name__ == "__main__":

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