#!/usr/bin/env python3
"""Answer Questions"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def answer_loop(reference):
    """Answers questions from a reference text:
        reference is the reference text.
        If the answer cannot be found in the reference text,
        respond with Sorry, I do not understand your question."""
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    while True:
        question = input('Q: ')
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        input_ids = tokenizer.encode(question, reference)
        input_mask = [1] * len(input_ids)
        input_type_ids = [0 if i < input_ids.index(
            102) else 1 for i in range(len(input_ids))]

        input_ids = tf.constant([input_ids])
        input_mask = tf.constant([input_mask])
        input_type_ids = tf.constant([input_type_ids])

        outputs = model([input_ids, input_mask, input_type_ids])
        start_index = tf.argmax(outputs[0][0][1:]) + 1
        end_index = tf.argmax(outputs[1][0][1:]) + 2

        answer_tokens = tokenizer.convert_ids_to_tokens(
            input_ids[0][start_index:end_index])
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        if answer:
            print('A: ' + answer)
        else:
            print('A: Sorry, I do not understand your question.')
