import torch

import torch.nn as nn

lstm = nn.lstm(
    input_size = 3,
    hidden_size = 5,
    batch_first=True
)