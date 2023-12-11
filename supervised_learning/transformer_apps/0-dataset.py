#!/usr/bin/env python3
"""Dataset class"""

def __init__(self):
    """Creates the instance attributes:
        data_train, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset train split, loaded as_supervided.
        data_valid, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset validate split, loaded as_supervided.
        tokenizer_pt is the Portuguese tokenizer created from the training set
        tokenizer_en is the English tokenizer created from the training set."""

def tokenize_dataset(self, data):
    """Creates sub-word tokenizers for our dataset:
        data is a tf.data.Dataset whose examples are formatted
            as a tuple (pt, en).
        pt is the tf.Tensor containing the Portuguese sentence.
        en is the tf.Tensor containing the corresponding English sentence.
        The maximum vocab size should be set to 2**15
        Returns: tokenizer_pt, tokenizer_en:
        tokenizer_pt is the Portuguese tokenizer.
        tokenizer_en is the English tokenizer."""
