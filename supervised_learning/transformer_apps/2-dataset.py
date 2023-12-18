#!/usr/bin/env python3
"""Dataset class"""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf


class Dataset:
    """Defines the Dataset class"""
    def __init__(self):
        """Creates the instance attributes:
            data_train, which contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset train split, loaded as_supervised.
            data_valid, which contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset validate split, loaded as_supervised.
            tokenizer_pt is the Portuguese tokenizer created from
                the training set.
            tokenizer_en is the English tokenizer created from
                the training set."""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # For tf_encode, update  attributes by tokenizing the examples.
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

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
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder\
            .build_from_corpus((en.numpy() for pt, en in data),
                               target_vocab_size=2**15)
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder\
            .build_from_corpus((pt.numpy() for pt, en in data),
                               target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation into tokens:
            pt is the tf.Tensor containing the Portuguese sentence.
            en is the tf.Tensor containing the corresponding English sentence.
            The tokenized sentences should include the start and end of
                sentence tokens.
            The start token should be indexed as vocab_size.
            The end token should be indexed as vocab_size + 1.
            Returns: pt_tokens, en_tokens
            pt_tokens is a np.ndarray containing the Portuguese tokens.
            en_tokens is a np.ndarray. containing the English tokens."""
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())

        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + \
            [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + \
            [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method.
            Set the shape of the pt and en return tensors."""
        pt, en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        pt.set_shape([None])
        en.set_shape([None])
        return pt, en
