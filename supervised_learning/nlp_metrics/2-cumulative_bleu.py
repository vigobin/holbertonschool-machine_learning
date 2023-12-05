#!/usr/bin/env python3
"""Cumulative N-gram BLEU score"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence:
        references is a list of reference translations.
        each reference translation is a list of the words in the translation.
        sentence is a list containing the model proposed sentence.
        n is the size of the largest n-gram to use for evaluation.
        All n-gram scores should be weighted evenly.
        Returns: the cumulative n-gram BLEU score."""
    sentence_ngrams = [tuple(sentence[i:i+n]) for i in range(
        len(sentence) - n+1)]
    sentence_counts = {
        word: sentence_ngrams.count(word) for word in sentence_ngrams}
    max_counts = {}

    for ref in references:
        ref_ngrams = [tuple(ref[i:i+n]) for i in range(
            len(ref) - n+1)]
        ref_counts = {word: ref_ngrams.count(word) for word in ref_ngrams}
        for word in ref_counts:
            max_counts[word] = max(max_counts.get(word, 0), ref_counts[word])

    clipped_counts = {word: min(count, max_counts.get(word, 0)) for word,
                      count in sentence_counts.items()}

    bleu_score = sum(clipped_counts.values()) / max(sum(
        sentence_counts.values()), 1)

    closest_ref_len = min(len(ref) for ref in references)

    if len(sentence) > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / len(sentence))

    return brevity_penalty * bleu_score
