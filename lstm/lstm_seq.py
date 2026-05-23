import numpy as np
from lstm import LSTMCell

class LSTM: 
    def __init__(self, input_size, hidden_size): 
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    
    def forward(self, sequence): 
        """
        Sequcen shape: 
        (seq_len, input_size)
        """
        seq_len = sequence.shape[0]

        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        outputs = []

        for t in range(seq_len): 
            xt = sequence[t].reshape(-1, 1)
            h, c = self.cell.forward(xt, h, c)
            outputs.append(h)
        
        return outputs

if __name__ == "__main__": 
    seq_len = 4
    input_size = 3
    hidden_size = 5

    sequence = np.random.rand(seq_len, input_size)
    model = LSTM(input_size, hidden_size)
    outputs = model.forward(sequence)

    for t, out in enumerate(outputs): 
        print(f"\nTime step {t}")
        print(out)