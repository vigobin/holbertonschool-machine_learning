#!/usr/bin/env python3
"""Extract Word2Vec"""

from keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.
        model is a trained gensim word2vec models
        Returns: the trainable keras Embedding."""
    vocab_size = len(model.wv.key_to_index) + 1
    vector_size = model.vector_size

    weight_matrix = np.zeros((vocab_size, vector_size))

    for word, i in model.wv.key_to_index.items():
        weight_matrix[i] = model.wv[word]

    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=vector_size,
                                weights=[weight_matrix],
                                trainable=True)

    return embedding_layer
