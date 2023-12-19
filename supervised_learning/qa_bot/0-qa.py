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
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    input_ids = tokenizer.encode(question, reference)
    input_mask = [1] * len(input_ids)
    input_type_ids = [0 if i < input_ids.index(102) else 1
                      for i in range(len(input_ids))]

    input_ids = tf.constant([input_ids])
    input_mask = tf.constant([input_mask])
    input_type_ids = tf.constant([input_type_ids])

    outputs = model([input_ids, input_mask, input_type_ids])
    start_index = tf.argmax(outputs[0][0][1:]) + 1
    end_index = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0][start_index:end_index])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer == '[CLS]' or answer == '[SEP]':
        return None

    return answer
