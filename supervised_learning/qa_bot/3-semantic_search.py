#!/usr/bin/env python3
"""Semantic Search"""

import os
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents:
        corpus_path is the path to the corpus of reference documents on which
            to perform semantic search.
        sentence is the sentence from which to perform semantic search.
        Returns: the reference text of the document most similar to
            sentence."""
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(model_url)

    corpus_sentences = []
    for filename in os.listdir(corpus_path):
        with open(os.path.join(
                corpus_path, filename), 'r', encoding='utf-8') as file:
            document_text = file.read()
            corpus_sentences.append(document_text)

    corpus_embeddings = embed(corpus_sentences)
    query_embedding = embed([sentence])

    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    most_similar_index = similarities.argmax()

    most_similar_document_path = os.path.join(corpus_path, os.listdir(
         corpus_path)[most_similar_index])

    with open(most_similar_document_path, 'r', encoding='utf-8') as file:
        most_similar_document_text = file.read()

    return most_similar_document_text
