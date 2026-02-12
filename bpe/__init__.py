"""
BPE (Byte Pair Encoding) tokenizer module.
"""
from .bpe import (
    tokenize,
    get_corpus,
    get_pair_counts,
    merge_pairs,
    train,
    encode,
    decode
)

__all__ = [
    'tokenize',
    'get_corpus',
    'get_pair_counts',
    'merge_pairs',
    'train',
    'encode',
    'decode'
]
