#!/usr/bin/env python3
"""N-gram BLEU score"""


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence:
        references is a list of reference translations.
        each reference translation is a list of the words in the translation.
        sentence is a list containing the model proposed sentence.
        n is the size of the n-gram to use for evaluation.
        Returns: the n-gram BLEU score."""
    