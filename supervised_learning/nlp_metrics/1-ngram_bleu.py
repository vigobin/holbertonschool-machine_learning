#!/usr/bin/env python3
"""N-gram BLEU score"""

import numpy as np


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence:
        references is a list of reference translations.
        each reference translation is a list of the words in the translation.
        sentence is a list containing the model proposed sentence.
        n is the size of the n-gram to use for evaluation.
        Returns: the n-gram BLEU score."""
    sentence_counts = {word: sentence.count(word) for word in sentence}
    max_counts = {}

    for ref in references:
        ref_counts = {word: ref.count(word) for word in ref}
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
