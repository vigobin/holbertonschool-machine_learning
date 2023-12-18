#!/usr/bin/env python3
"""Question Answering"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to answer
        a question:
        question is a string containing the question to answer.
        reference is a string containing the reference document from which to
            find the answer.
        Returns: a string containing the answer.
        If no answer is found, return None.
        Your function should use the bert-uncased-tf2-qa model from the
            tensorflow-hub library.
        Your function should use the pre-trained BertTokenizer,
            bert-large-uncased-whole-word-masking-finetuned-squad,
            from the transformers library."""
