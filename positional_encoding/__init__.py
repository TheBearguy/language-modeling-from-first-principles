"""
Position-aware language model module (learned and sinusoidal positions).
"""
from .lm_position_encoding import (
    init_learned_positions,
    init_sinusoidal_positions,
    lm_with_positions,
    train_language_model_with_positions
)

__all__ = [
    'init_learned_positions',
    'init_sinusoidal_positions',
    'lm_with_positions',
    'train_language_model_with_positions'
]
