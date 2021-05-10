'''
File: plotters.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports 
import tensorflow as tf
import numpy as np

def score_file_writer(true_y_batch, y_batch_train, fixed_score, plus):
    # The noisified score
    noisified_score = tf.reduce_sum(tf.cast(tf.equal(true_y_batch + y_batch_train, 1), tf.int32))
    # Calculate the flipping loss; the low the better
    score_loss = noisified_score - fixed_score
    actual_score = tf.logical_and(tf.equal(plus,1), tf.equal(true_y_batch+y_batch_train,1))
    actual_score = tf.reduce_sum(tf.cast(actual_score, tf.int32))
    score_file = open("score_file.txt","a")
    score_file.write(f'Noised: {str(noisified_score.numpy())} difFromTru: {str(fixed_score.numpy())}  Loss: {str(score_loss.numpy())} Fixed: {str(actual_score.numpy())} \n')
    score_file.close()

