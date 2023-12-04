#!/usr/bin/env python3
"""Unigram BLEU score"""


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence:
        references is a list of reference translations.
        each reference translation is a list of the words in the translation.
        sentence is a list containing the model proposed sentence.
        Returns: the unigram BLEU score."""
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

    return bleu_score
