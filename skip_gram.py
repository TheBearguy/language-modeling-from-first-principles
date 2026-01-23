import numpy as np
from bpe import encode, merge_rules, text, vocab_to_id

from bpe import vocab

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


def train_skipgram(
    token_ids, 
    vocab_size, 
    dim = 32, 
    window_size = 2, 
    neg_samples = 5, 
    lr = 0.05, 
    epochs = 10
): 
    E, O = init_embeddings(vocab_size, dim)
    pairs = build_skipgram_pairs(token_ids, window_size)

    for epoch in range(epochs): 
        np.random.shuffle(pairs)

        for center, context in pairs: 
            negatives = samples_negative(vocab_size, neg_samples)
            skipgram_step(E, O, negatives, center, context, lr)
        print(f"\nepoch {epoch + 1} complete")
    return E

if __name__ == "__main__": 
    token_ids, _ = encode(text, merge_rules, vocab_to_id)
    E = train_skipgram(
        token_ids, 
        vocab_size = len(vocab_to_id), 
        dim=32
    )