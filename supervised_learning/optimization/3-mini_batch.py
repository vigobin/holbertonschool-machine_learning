#!/usr/bin/env python3
"""Mini Batch"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using mini-batch
    gradient descent"""
    m = X_train.shape[0]
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

    for epoch in range(epochs + 1):
        print("After {} epochs:".format(epoch))
        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        print("\tTraining Cost: {}".format(train_cost))
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        print("\tTraining Accuracy: {}".format(train_accuracy))
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        print("\tValidation Cost: {}".format(valid_cost))
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        print("\tValidation Accuracy: {}".format(valid_accuracy))
        if epoch == epochs:
            break
        X_train_s, Y_train_s = shuffle_data(X_train, Y_train)
        if (m % batch_size) is 0:
            mini_batch_total = m // batch_size
        else:
            mini_batch_total = (m // batch_size) + 1

        step_number = 0
        for mini_batch in range(mini_batch_total):
            low = mini_batch * batch_size
            high = ((mini_batch + 1) * batch_size)
            if high > m:
                high = m
            sess.run(train_op, feed_dict={x: X_train_s[low:high, :],
                                          y: Y_train_s[low:high, :]})
            step_number += 1
            if (step_number % 100) is 0:
                print("\tStep {}:".format(step_number))
                step_cost = sess.run(
                    loss,
                    feed_dict={x: X_train_s[low:high, :],
                               y: Y_train_s[low:high, :]})
                print("\t\tCost: {}".format(step_cost))
                step_accuracy = sess.run(
                    accuracy,
                    feed_dict={x: X_train_s[low:high, :],
                               y: Y_train_s[low:high, :]})
                print("\t\tAccuracy: {}".format(step_accuracy))
        return saver.save(sess, save_path)
