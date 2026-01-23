import numpy as np

from bpe.bpe import target
from language_model.language_model import softmax

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