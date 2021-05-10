'''
File: loss_fun.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports
import tensorflow as tf
from tensorflow import keras
import scipy
import scipy.stats
from scipy.stats import wasserstein_distance as wasserstein

# Local imports
from modules.mmd import mmd
from modules.groupLasso import groupLasso, choose_noisy_classes, flip_classes

def loss_fun(y_batch_train, logits_1, logits_2, batch_size, l2_logits_m1, l2_logits_m2, 
        sigma, swap, swap_rate, lambda2, lambda3, flip_bound, start_flipping, 
        flip_per, miss_alpha, extra_beta, divergence_metric, alpha, pred_treshold):
    
    '''
    Group Lasso
    Error loss array is the average of group lassos of extra and missing class labels.
    noisy_sample is the sample with the highest loss within the minibatch.
    noisy_class is the class of the noisy_sample that is gonna be flipped.
    '''
    sigmoided_logits_1 = tf.math.sigmoid(logits_1)
    sigmoided_logits_2 = tf.math.sigmoid(logits_2)
    accuracy_logits_1 = tf.cast(sigmoided_logits_1 >= pred_treshold, tf.float32)
    accuracy_logits_2 = tf.cast(sigmoided_logits_2 >= pred_treshold, tf.float32)
    error_loss_array_1, classes_1 = groupLasso(y_batch_train, accuracy_logits_1, miss_alpha, extra_beta)
    error_loss_array_2, classes_2 = groupLasso(y_batch_train, accuracy_logits_2, miss_alpha, extra_beta)

    if start_flipping > flip_bound:

        '''
        Flip the labels
        To flip the labels both of the networks must be in consensus
        Get the noisy samples with corresponding classes from the mini batch.
        '''
        noisy_samples, noisy_classes = choose_noisy_classes(error_loss_array_1, error_loss_array_2, classes_1, classes_2, flip_per)
        
        # Flip classes
        unflipped_y_batch_train = y_batch_train
        y_batch_train = flip_classes(y_batch_train, noisy_samples, noisy_classes)
        
        # Calculate the loss arrays again.
        error_loss_array_1, classes_1 = groupLasso(y_batch_train, accuracy_logits_1, miss_alpha, extra_beta)
        error_loss_array_2, classes_2 = groupLasso(y_batch_train, accuracy_logits_1, miss_alpha, extra_beta)
    
    loss_array_1 = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(y_batch_train, logits_1, 1.0), axis=1)
    loss_array_2 = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(y_batch_train, logits_2, 1.0), axis=1)

    brutto_loss_array_1 = loss_array_1 + alpha * error_loss_array_1
    brutto_loss_array_2 = loss_array_2 + alpha * error_loss_array_2
    
    if divergence_metric == 'mmd':
        L2 = mmd(l2_logits_m1, l2_logits_m2, sigma) * lambda2
        L3 = mmd(logits_1, logits_2, sigma) * lambda3
    elif divergence_metric == 'shannon':
        kl = keras.losses.KLDivergence()
        M2 = (0.5) * (l2_logits_m1 + l2_logits_m2)
        M3 = (0.5) * (logits_1 + logits_2)
        L2 = lambda2 * (0.5 * kl(l2_logits_m1, M2) + 0.5 * kl(l2_logits_m2, M2))
        L3 = lambda3 * (0.5 * kl(logits_1, M3) + 0.5 * kl(logits_2, M3))
    elif divergence_metric == 'wasserstein':
        L2 = wasserstein(tf.reshape(l2_logits_m1,-1), tf.reshape(l2_logits_m2,-1)) * lambda2
        L3 = wasserstein(tf.reshape(logits_1,-1), tf.reshape(logits_2,-1)) * lambda3
    elif divergence_metric == 'nothing':
        L2 = 0
        L3 = 0
    
    # Chooses the args of the (batch_size*swap_rate) low loss samples in the corresponding low_loss arrays
    low_loss_args_1 = tf.argsort(brutto_loss_array_1)[:int(batch_size * swap_rate)]
    low_loss_args_2 = tf.argsort(brutto_loss_array_2)[:int(batch_size * swap_rate)]

    if swap == 1:
        # Gets the low_loss_samples as conducted by the peer network
        low_loss_samples_1 = tf.gather(loss_array_1, low_loss_args_2)
        low_loss_samples_2 = tf.gather(loss_array_2, low_loss_args_1)
    elif swap == 0:
        # No swap 
        low_loss_samples_1 = tf.gather(loss_array_1, low_loss_args_1)
        low_loss_samples_2 = tf.gather(loss_array_2, low_loss_args_2)
    
    loss_1 = tf.nn.compute_average_loss(low_loss_samples_1, global_batch_size=int(batch_size * swap_rate))
    loss_2 = tf.nn.compute_average_loss(low_loss_samples_2, global_batch_size=int(batch_size * swap_rate))

    return loss_1+L3-L2, loss_2+L3-L2, L3, L2

