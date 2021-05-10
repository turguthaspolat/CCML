'''
File: groupLasso.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de) 
'''

# Standard library imports
import math

# Third party imports 
import numpy as np
import tensorflow as tf


@tf.function
def groupLasso(y, p, alpha, beta):

    '''
    input: y = labels. (batch_size, num_of_classes)
           p = predictions. (batch_size, num_of_classes)
           alpha = factor of loss_miss on the error_loss. float between 0 and 1
           beta = factor of extra_miss on the error_loss. float between 0 and 1 
    
    output: losses = the tensor that has (batch_size) errors for each sample.
            classes =  the tensor that has the index of the potential noisy class for each sample.

    Group Lasso Error for missing classes: E(zero_classes) sqrt( E(one_classes) F(K,L)^2 )
    Group Lasso Error for extra classes: E(one_classes) sqrt( E(zero_classes) F(K,L)^2 )
    E(x) is the cumulative sum!
    THE DISCRIMINATIVE FUNCTION IS F(K,L) = MAX(0, 2*(L-K)+1)
    ARTIFICIAL TRESHHOLD IS 0.5 FOR BOTH OF THEM
    For the true classification, the function gives -1
    For a missing class or an extra class, the function gives +1
    And for the extreme, which is missing class and extra class together, the function gives +3

    How does the algorithm chooses the noisy class of every sample?
    The algorithm accepts the class with highest loss as noisy class for each sample.

    Caveat:
    The losses are not only a measure of how noisy each sample is.
    They are also a measure of how good the neural network is doing.
    Hence, evaluating the loss as only a measure of noise in the sample would be erroneous.
    When the pairwise difference of predictions between every positive and negative classes
    is bigger than 0.5, which is the mentioned natural treshhold, the algorithm only measures
    how noisy a sample is. That is why in order to flip classes, the network must be stable.
    So the best way to avoid wrong flips is to wait a couple of epochs, then flip.
    '''

    # Invert labels of batch
    inverted_y = tf.math.add(tf.math.multiply(y, -1.), 1.)
    
    # I only want to do pairwise computation between 0 and 1 labels. This eliminates the 1-1 and 0-0 pairs.
    select_examples = tf.cast(tf.math.multiply(tf.expand_dims(y, 1), tf.expand_dims(inverted_y, 2)), tf.float32)

    # Calculate the lasso errors
    k = tf.math.subtract(tf.expand_dims(p, 2), tf.expand_dims(p, 1))
    k = tf.math.multiply(k, 2.)
    k = tf.math.add(k, 1.)
    k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
    errors = tf.math.square(k)
    
    # Select instances: Zero the 1-1 and 0-0 pairs out
    errors = tf.math.multiply(errors, select_examples)

    # Calculate the loss_miss
    k = tf.reduce_sum(errors, 2)
    # Escape instability when taking the gradient of any 0 values with tf.math.sqrt
    k = k + 1e-10
    groups_miss = tf.math.sqrt(k)
    loss_miss = tf.reduce_sum(groups_miss, axis=1)

    # Choose the potential missing class label for every sample
    potential_missing = tf.argmax(groups_miss, axis=1)
    missing_high = tf.reduce_max(groups_miss, axis=1)
    
    # Calculate the loss_extra
    k = tf.reduce_sum(errors, 1)
    # Escape instability when taking the gradient of any 0 values with tf.math.sqrt
    k = k + 1e-10
    groups_extra = tf.math.sqrt(k)
    loss_extra = tf.reduce_sum(groups_extra, 1)

    # Choose the potential extra label for every sample
    potential_extra = tf.argmax(groups_extra, 1)
    extra_high = tf.reduce_max(groups_extra, 1)

    # Choose the missing or the extra per sample
    classes = tf.where(missing_high > extra_high, potential_missing, potential_extra)

    # Enter the loss for the sample                                             
    losses = tf.math.add(tf.math.multiply(alpha, loss_miss), tf.math.multiply(beta, loss_extra))

    return losses, classes

def choose_noisy_classes(losses1, losses2, classes1, classes2, flip_per):
    
    # Create an index tensor to map back to sample indexes
    indices = tf.range(len(classes1))

    # Filter out the classes that are not in consensus
    class_filter = tf.logical_not(tf.cast(classes1 - classes2, tf.bool))

    filtered_losses1 = tf.boolean_mask(losses1, class_filter)
    filtered_losses2 = tf.boolean_mask(losses2, class_filter)
    # Get the average losses to choose from
    avg_losses = (filtered_losses1 + filtered_losses2) / 2.0

    # Filter the classes
    filtered_classes1 = tf.boolean_mask(classes1, class_filter)

    # Filter the indices accordingly
    filtered_indices = tf.boolean_mask(indices, class_filter)

    # Calculate number of samples to flip
    num_flips = math.ceil(len(filtered_classes1) * flip_per)

    # Choose the sample with the highest loss within the batch; flip its class
    highest_losses, noisy_sample = tf.math.top_k(avg_losses, k=num_flips)
    noisy_class = tf.gather(filtered_classes1, noisy_sample)
    noisy_sample = tf.gather(filtered_indices, noisy_sample)
    
    return noisy_sample, tf.cast(noisy_class, tf.int32)

def flip_classes(y, noisy_samples, noisy_classes):
    stacked = tf.stack([noisy_samples, noisy_classes])
    y_np = tf.make_ndarray(tf.make_tensor_proto(y))
    y_np[tuple(stacked)] = -y_np[tuple(stacked)] + 1
    flipped_y = tf.constant(y_np)
    
    return flipped_y

