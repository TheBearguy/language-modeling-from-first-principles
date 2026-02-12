"""
RNN language model module with BPTT.
"""
from .rnn import (
    init_rnn_params,
    init_learned_positions,
    init_sinusoidal_positions,
    rnn_lm_forward,
    rnn_lm_backward,
    train_rnn_language_model
)

__all__ = [
    'init_rnn_params',
    'init_learned_positions',
    'init_sinusoidal_positions',
    'rnn_lm_forward',
    'rnn_lm_backward',
    'train_rnn_language_model'
]
