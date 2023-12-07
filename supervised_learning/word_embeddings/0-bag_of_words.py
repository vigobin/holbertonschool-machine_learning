#!/usr/bin/env python3
"""Bag Of Words"""

import numpy as np
import string


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix.
        sentences is a list of sentences to analyze.
        vocab is a list of the vocabulary words to use for the analysis.
        If None, all words within sentences should be used.
        Returns: embeddings, features
            embeddings is a numpy.ndarray of shape (s, f)
            containing the embeddings.
            s is the number of sentences in sentences.
            f is the number of features analyzed.
            features is a list of the features used for embeddings."""
    sentences = [sentence.lower().translate(str.maketrans(
        '', '', string.punctuation)) for sentence in sentences]

    if vocab is None:
        vocab = sorted(set(
            word for sentence in sentences for word in sentence.split()))

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for j, word in enumerate(vocab):
            embeddings[i, j] = words.count(word)

    return embeddings, list(vocab)
