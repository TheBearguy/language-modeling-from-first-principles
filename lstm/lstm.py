import numpy as np

def sigmoid(x): 
    return (1 / (1 + np.exp(-x)))

def softmax(x): 
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

def one_hot(index, vocab_size): 
    vec = np.zeros((vocab_size, 1))
    vec[index] = 1

    return vec

class LSTMCell:
    def __init__(self, input_size, hidden_size, output_size): 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # combined input size
        z_size = input_size + hidden_size

        # Forget gate: 
        self.Wf = np.random.randn(hidden_size, z_size) * 0.1
        # Why random: 
        # Each neuron starts with:
        # random behavior, but small influence
        # Training later adjusts these weights.
        # If all weights started as zeros: every neuron would learn the exact same thing
        # symmetry problem.
        # Random initialization breaks symmetry.
        self.bf = np.zeros((hidden_size, 1))
        # Biases do NOT suffer from symmetry issues the same way weights do.
        # So initializing them to zero is standard.
        

        # Input gate 
        self.Wi = np.random.randn(hidden_size, z_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))

        # Candidate memory
        self.Wc = np.random.randn(hidden_size, z_size) * 0.1
        self.bc = np.zeros((hidden_size, 1))

        # Output gate
        self.Wo = np.random.randn(hidden_size, z_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))

        # prediction 
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    
    def forward(self, xt, h_prev, c_prev): 
        """
        xt: (input_size, 1)
        h_prev: (hidden_size, 1)
        c_prev: (hidden_size, 1)
        """

        # concatenate hidden + input
        z = np.vstack((h_prev, xt))

        # Forget gate 
        f_t = sigmoid(self.Wf @ z + self.bf)

        # Input gate 
        i_t = sigmoid(self.Wi @ z + self.bi)

        # candidate memory 
        c_hat = np.tanh(self.Wc @ z + self.bc)

        # Update cell state
        c_t = f_t * c_prev + i_t * c_hat

        # Output gate
        o_t = sigmoid(self.Wo @ z + self.bo)

        # hidden state
        h_t = o_t * np.tanh(c_t)

        # Prediction
        y_t = self.Wy @ h_t + self.by
        pt = softmax(y_t) # Probabilities

        return h_t, c_t, y_t, pt
    

if __name__ == "__main__": 
    text = "hello"
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)

    char_to_idx = {c:i for i, c in enumerate(vocab)}
    idx_to_char = {i:c for i, c in enumerate(vocab)}

    print("Vocabulary:", vocab)
    print("Char to idx:", char_to_idx)

    input_size = vocab_size
    hidden_size = 8
    output_size = vocab_size

    lstm = LSTMCell(input_size, hidden_size, output_size)
    
    # Example training pair
    # h -> e

    input_char = 'h'
    target_char = 'e'

    input_idx = char_to_idx[input_char]
    target_idx = char_to_idx[target_char]

    # one hot input char
    xt = one_hot(input_idx, vocab_size)
    # xt = np.random.randn(input_size, 1)
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))

    h_t, c_t, y_t, pt = lstm.forward(xt, h_prev, c_prev)

    # Cross Entropy loss
    loss = -np.log(pt[target_idx, 0])

    print("\nInput character:")
    print(input_char)

    print("\nTarget character:")
    print(target_char)

    print("\nPredicted probabilities:")

    for i in range(vocab_size): 
        ch = idx_to_char[i]
        prob = pt[i, 0]

        print(f"{ch}: {prob:.4f}")
    
    print("\nLoss:")
    print(loss)