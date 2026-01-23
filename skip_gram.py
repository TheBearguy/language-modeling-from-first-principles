import numpy as np

def build_skipgram_pairs(token_ids, window_size = 2): 
    pairs = []
    n = len(token_ids)
    for i in range(n): 
        center = token_ids[i]
        for j in range( max(0, i - window_size), min(n, i + window_size + 1)): 
            if i != j: 
                pairs.append((center, token_ids[j]))
    return pairs

def init_embeddings(vocab_size, dim, seed = 42): 
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.01, (vocab_size, dim))
    O = rng.normal(0, 0.01, (vocab_size, dim))
    # Rows = Tokens
    # Columns = Free coordinates
    return E, O

def samples_negative(vocab_size, k): 
    return np.random.randint(0, vocab_size, size = k)

def sigmoid(score):
    return 1 / (1 + np.exp(-score))

def skipgram_step(E, O, negatives, center, context, lr): 
    v_w = E[center]
    v_c = E[context]

    score = np.dot(v_w, v_c)

    #probability
    p = sigmoid(score)

    gradient = p-1

    E[center]= E[center]- lr * gradient * v_c
    O[context] = O[context] - lr * gradient * v_w

    # negative samples: 
    for neg in negatives: 
        v_n = O[neg]
        score_n = np.dot(v_w, v_n)
        p_n = sigmoid(score_n)
        E[center] = E[center] - lr * p_n * v_n  
        O[context] = O[context] - lr * p_n * v_w