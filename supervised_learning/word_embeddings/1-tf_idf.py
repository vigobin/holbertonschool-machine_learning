#!/usr/bin/env python3
"""TF-IDF function"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding
        sentences is a list of sentences to analyze.
        vocab is a list of the vocabulary words to use for the analysis.
        If None, all words within sentences should be used.
        Returns: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
        s is the number of sentences in sentences.
        f is the number of features analyzed.
        features is a list of the features used for embeddings."""
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()

    features = sorted(list(set(vectorizer.get_feature_names_out())))

    return embeddings, features
