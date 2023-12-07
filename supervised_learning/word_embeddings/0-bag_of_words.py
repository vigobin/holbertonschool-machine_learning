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
        # Only include the singular form of a word if both singular
        # and plural forms are present
        vocab = [word for word in vocab if not (
            word.endswith('s') and word[:-1] in vocab)]

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for j, word in enumerate(vocab):
            count = words.count(word)
            if word+'s' in words:
                count += words.count(word+'s')
            embeddings[i, j] = count

    return embeddings, vocab
