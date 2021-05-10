import numpy as np
import tensorflow as tf
from groupLasso import groupLasso
from timeit import default_timer as timer

def test():
    labels = tf.constant([[1,0,1,0,1,1,0,0,1],[0,1,1,0,1,0,0,1,1]])
    labels_float = tf.constant([[1.,0.,1.,0.,1.,1.,0.,0.,1.],[0.,1.,1.,0.,1.,0.,0.,1.,1.]])
    labels2 = tf.constant([[1.,0.,0.,0.,1.,1.,0.,1.,1.]])
    preds2 = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]])
    

    noisy_labels_float = tf.constant([[1.,0.,0.,0.,1.,1.,0.,0.,1.],[0.,1.,1.,0.,1.,0.,0.,1.,1.],[1.,0.,0.,0.,1.,1.,0.,0.,1.],[0.,1.,1.,0.,1.,0.,0.,1.,1.]]) # Missing class at [0][2]
    preds = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.1,0.8,0.8,0.2,0.9,0.2,0.2,0.9,0.9],[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.1,0.8,0.8,0.2,0.9,0.2,0.2,0.9,0.9]])
    
    loss_array = tf.nn.sigmoid_cross_entropy_with_logits(labels=noisy_labels_float, logits=preds)
    loss_array = tf.math.reduce_mean(loss_array, axis=1)
    start = timer()
    losses, classes = groupLasso(tf.make_ndarray(tf.make_tensor_proto(noisy_labels_float)), tf.make_ndarray(tf.make_tensor_proto(preds)))
    # losses, classes = groupLasso(noisy_labels_float, preds)
    end = timer()
    print(f"end - start: {end - start}")

    print(f"loss_array: {loss_array}")
    print(f"losses: {losses}")
    print(f"classes: {classes}")

    print(f"type(losses): {type(losses)}")

    losses = 0.2 * np.array(losses)
    print(f"losses: {losses}")

#test()



def test2():
    labels = tf.constant([[1,0,1,0,1,1,0,0,1],[0,1,1,0,1,0,0,1,1]])
    labels_float = tf.constant([[1.,0.,1.,0.,1.,1.,0.,0.,1.],[0.,1.,1.,0.,1.,0.,0.,1.,1.]])
    labels2 = tf.constant([[1.,0.,0.,0.,1.,1.,0.,1.,1.]])
    preds2 = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]])

    noisy_labels_float = tf.constant([[1.,0.,1.,0.,1.,1.,0.,0.,1.],[0.,1.,1.,0.,1.,0.,0.,1.,1.],[1.,0.,0.,0.,1.,1.,0.,0.,1.],[0.,1.,1.,0.,1.,0.,0.,1.,1.]]) # Missing class at [0][2]
    preds = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.1,0.8,0.8,0.2,0.9,0.2,0.2,0.9,0.9],[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.1,0.8,0.8,0.2,0.9,0.2,0.2,0.9,0.9]])
    
    double_noisy_y1 = np.array([[1,0,0,0,1,1,0,1,1],[1,0,0,0,1,1,0,1,1]]) # Missing class at index 2 and extra class at index 7
    p4 = np.array([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.78,0.34,0.72,0.2,0.86,0.62,0.4,0.17,0.95]])
    
    '''
    # Gives me the corresponding sparse tensors
    ones_bools = tf.cast(noisy_labels_float, tf.bool)
    zeros_bools = tf.math.logical_not(ones_bools)
    ones_indices = tf.where(ones_bools)
    zeros_indices = tf.where(zeros_bools)

    ones = tf.boolean_mask(preds, ones_bools)
    zeros = tf.boolean_mask(preds, zeros_bools)
    
    sparse_ones = tf.sparse.SparseTensor(indices=ones_indices, values=ones, dense_shape=preds.shape)
    sparse_zeros = tf.sparse.SparseTensor(indices=zeros_indices, values=zeros, dense_shape=preds.shape)
    '''

    '''
    # Get the ones and zeros, there are zeros inbetween
    inverted_labels = tf.cast(tf.math.logical_not(tf.cast(noisy_labels_float, tf.bool)), tf.float32)
    zeros = noisy_labels_float * preds
    print(f"zeros: {zeros}")

    ones = inverted_labels * preds
    print(f"ones: {ones}")
    '''

    '''
    # Get the ones and zeros, basically same as the one before
    num_nons = tf.math.count_nonzero(noisy_labels_float, axis=1)
    ones = tf.boolean_mask(preds, tf.cast(noisy_labels_float, tf.bool))
    ones_list = tf.split(ones, num_nons)
    stacked_ones = tf.ragged.stack(ones_list)
    
    num_zeros = len(noisy_labels_float[0]) - num_nons
    inverted_labels_bool = tf.math.logical_not(tf.cast(noisy_labels_float, tf.bool))
    zeros = tf.boolean_mask(preds, inverted_labels_bool)
    zeros_list = tf.split(zeros, num_zeros)
    stacked_zeros = tf.ragged.stack(zeros_list)

    #print(f"ones: {ones}")
    #print(f"ones_list: {ones_list}")
    print(f"stacked_ones: {stacked_ones}")
    #print(f"zeros: {zeros}")
    #print(f"zeros_list: {zeros_list}")
    print(f"stacked_zeros: {stacked_zeros}")
    '''
    
    # Get the ones and zeros as ragged tensors
    bool_y = tf.cast(double_noisy_y1, tf.bool)
    ones = tf.ragged.boolean_mask(p4, bool_y)
    
    inverted_labels_bool = tf.math.logical_not(bool_y)
    zeros = tf.ragged.boolean_mask(p4, inverted_labels_bool)

    # Stack corresponding ones and zeros on top of each other
    one_zero_pairs = tf.ragged.stack([ones,zeros], axis=1)

    LOSSES = []
    CLASSES = []

    # Fix this if you can!
    for one_zero_pair in one_zero_pairs:
        # Calculate missing class errors
        onez = tf.expand_dims(one_zero_pair[0], axis=0)
        zeroz = tf.expand_dims(one_zero_pair[1], axis=1)
        k = tf.math.subtract(zeroz, onez)
        k = tf.math.multiply(k, 2)
        k = tf.math.add(k, 1)
        k = tf.math.maximum(0, k)
        errors_miss = tf.math.square(k)

        # Calculate extra class errors
        onez = tf.expand_dims(one_zero_pair[0], axis=1)
        zeroz = tf.expand_dims(one_zero_pair[1], axis=0)
        k = tf.math.subtract(zeroz, onez)
        k = tf.math.multiply(k, 2)
        k = tf.math.add(k, 1)
        k = tf.math.maximum(0, k)
        errors_extra = tf.math.square(k)
        
        # Calculate the loss_miss
        k = tf.reduce_sum(errors_miss, axis=1)
        groups_miss = tf.math.sqrt(k)
        loss_miss = tf.reduce_sum(groups_miss)
        # Find the noisy missing class, choose the one with the biggest loss
        # Index that I get is not the real index of the missing class. It is the index of it in its own group, which is zeros!
        row_wise_min = tf.reduce_min(errors_miss, axis=1)
        indices = tf.where(tf.math.not_equal(row_wise_min, 0))
        potential_missing_classes = tf.gather(groups_miss, indices)
        try:
            potential_missing = tf.math.argmax(potential_missing_classes)
            missing_class_index = tf.gather(indices, potential_missing)
        except:
            missing_class_index = None

        # Calculate the extra_miss
        k = tf.reduce_sum(errors_extra, axis=1)
        groups_extra = tf.math.sqrt(k)
        loss_extra = tf.reduce_sum(groups_extra)
        # Find the noisy extra class, choose the one with the biggest loss
        # Index that I get is not the real index of the missing class. It is the index of it in its own group, which is zeros!
        row_wise_min = tf.reduce_min(errors_extra, axis=1)
        indices = tf.where(tf.math.not_equal(row_wise_min, 0))
        potential_extra_classes = tf.gather(groups_extra, indices)
        try:
            potential_extra = tf.math.argmax(potential_extra_classes)
            extra_class_index = tf.gather(indices, potential_extra)
        except:
            extra_class_index = None

        # Enter the loss for the sample
        LOSSES.append((loss_miss + loss_extra) / 2.0)
        # CLASSES.append()

        #break

    LOSSES = tf.cast(LOSSES, tf.float32)
    print(LOSSES)


# test2()

