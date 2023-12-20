#!/usr/bin/env python3
"""Semantic Search"""

import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents:
        corpus_path is the path to the corpus of reference documents on which
            to perform semantic search.
        sentence is the sentence from which to perform semantic search.
        Returns: the reference text of the document most similar to
            sentence."""
    corpus_sentences = []
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(os.path.join(
                    corpus_path, filename), 'r', encoding='utf-8') as file:
                corpus_sentences.append(file.read())

    corpus_sentences.append(sentence)

    vectorizer = TfidfVectorizer()
    corpus_embeddings = vectorizer.fit_transform(corpus_sentences)

    similarities = cosine_similarity(
        corpus_embeddings[-1:], corpus_embeddings[:-1])

    most_similar_index = similarities.argmax()

    return corpus_sentences[most_similar_index]
