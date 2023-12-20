#!/usr/bin/env python3
"""Multi-reference Question Answering"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

import os
from sklearn.metrics.pairwise import cosine_similarity


def question_answer(corpus_path):
    """Answers questions from multiple reference texts:
        corpus_path is the path to the corpus of reference documents."""
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    while True:
        question = input('Q: ')
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
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
