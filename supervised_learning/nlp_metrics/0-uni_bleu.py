#!/usr/bin/env python3
"""Unigram BLEU score"""

from collections import Counter
from nltk.util import ngrams
import math


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence:
        references is a list of reference translations.
        each reference translation is a list of the words in the translation.
        sentence is a list containing the model proposed sentence.
        Returns: the unigram BLEU score."""
    sentence_counts = Counter(ngrams(sentence, 1))
    max_counts = {}

    for ref in references:
        ref_counts = Counter(ngrams(ref, 1))
        for ngram in ref_counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    ref_counts[ngram])

    clipped_counts = {ngram: min(count, max_counts.get(ngram, 0)) for ngram,
                      count in sentence_counts.items()}

    bleu_score = sum(clipped_counts.values()) / max(sum(
        sentence_counts.values()), 1)

    return bleu_score
