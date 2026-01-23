from syslog import LOG_SYSLOG
import numpy as np


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