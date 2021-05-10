import numpy as np
import tensorflow as tf
import math

@tf.function
def groupLasso(y,p):

    inverted_y = -y + 1
    
    select_examples = tf.cast(tf.expand_dims(y, 1) * tf.expand_dims(inverted_y, 2), tf.float32)

    k = tf.subtract(tf.expand_dims(p, 2), tf.expand_dims(p, 1))
    k = tf.math.multiply(k, 2)
    k = tf.math.add(k, 1)
    k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
    errors = tf.math.square(k)
    
    # Select instances
    errors = errors * select_examples

    # Calculate the loss_miss
    k = tf.reduce_sum(errors, 2)
    groups_miss = tf.math.sqrt(k)
    loss_miss = tf.reduce_sum(groups_miss, axis=1)

    # Choose the potential missing
    potential_missing = tf.argmax(groups_miss, axis=1)
    missing_high = tf.reduce_max(groups_miss, axis=1)
    
    # Calculate the loss_extra
    k = tf.reduce_sum(errors, 1)
    groups_extra = tf.math.sqrt(k)
    loss_extra = tf.reduce_sum(groups_extra, 1)

    # Choose potential extra
    potential_extra = tf.argmax(groups_extra, 1)
    extra_high = tf.reduce_max(groups_extra, 1)

    # Choose the missing or the extra
    CLASSES = tf.where(missing_high > extra_high, potential_missing, potential_extra)

    # Enter the loss for the sample                                             
    loss = (loss_miss + loss_extra) / 2.0

    return loss, CLASSES

#@tf.function
def noForLasso(y,p):
    # Get the ones and zeros as ragged tensors
    
    bool_y = tf.cast(y, tf.bool)
    ones = tf.ragged.boolean_mask(p, bool_y)

    inverted_labels_bool = tf.math.logical_not(bool_y)
    zeros = tf.ragged.boolean_mask(p, inverted_labels_bool)

    # Stack corresponding ones and zeros on top of each other
    one_zero_pairs = tf.ragged.stack([ones,zeros], axis=1)

    # print(one_zero_pairs.shape)
    # print(one_zero_pairs)

    tf.RaggedTensor

    LOSSES = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    CLASSES = []
    
    #print(tf.subtract(tf.expand_dims(ones, axis=1), tf.expand_dims(zeros, axis=0)))
    #print(tf.subtract(tf.expand_dims(ones[0], axis=1), tf.expand_dims(zeros[0], axis=0)))

    # print(tf.subtract(tf.expand_dims(one_zero_pairs[:, 0], axis=0), tf.expand_dims(one_zero_pairs[:,1], axis=1)))
    # print(tf.subtract(tf.expand_dims(zeros, axis=2), tf.expand_dims(ones, axis=0)).shape)
    # print(tf.subtract(tf.expand_dims(zeros, axis=1), tf.expand_dims(ones, axis=0)).shape)

    '''
    # Calculate missing class errors
    onez = tf.expand_dims(one_zero_pair[0], axis=0)
    zeroz = tf.expand_dims(one_zero_pair[1], axis=1)
    k = tf.math.subtract(zeroz, onez)
    k = tf.math.multiply(k, 2)
    k = tf.math.add(k, 1)
    k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
    errors_miss = tf.math.square(k)

    # Calculate extra class errors
    onez = tf.expand_dims(one_zero_pair[0], axis=1)
    zeroz = tf.expand_dims(one_zero_pair[1], axis=0)
    k = tf.math.subtract(zeroz, onez)
    k = tf.math.multiply(k, 2)
    k = tf.math.add(k, 1)
    k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
    errors_extra = tf.math.square(k)

    # Calculate the loss_miss
    k = tf.reduce_sum(errors_miss, axis=1)
    groups_miss = tf.math.sqrt(k)
    loss_miss = tf.reduce_sum(groups_miss)

    # Calculate the extra_miss
    k = tf.reduce_sum(errors_extra, axis=1)
    groups_extra = tf.math.sqrt(k)
    loss_extra = tf.reduce_sum(groups_extra)

    # Enter the loss for the sample
    loss = (loss_miss + loss_extra) / 2.0
    '''

    return LOSSES, CLASSES

def calc_errors(one_zero_pairs):
    # Calculate missing class errors
    onez = tf.expand_dims(one_zero_pairs[0], axis=0)
    zeroz = tf.expand_dims(one_zero_pairs[1], axis=1)
    k = tf.math.subtract(zeroz, onez)
    k = tf.math.multiply(k, 2)
    k = tf.math.add(k, 1)
    k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
    errors_miss = tf.math.square(k)

    # Calculate extra class errors
    onez = tf.expand_dims(one_zero_pairs[0], axis=1)
    zeroz = tf.expand_dims(one_zero_pairs[1], axis=0)
    k = tf.math.subtract(zeroz, onez)
    k = tf.math.multiply(k, 2)
    k = tf.math.add(k, 1)
    k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
    errors_extra = tf.math.square(k)

    # Calculate the loss_miss
    k = tf.reduce_sum(errors_miss, axis=1)
    groups_miss = tf.math.sqrt(k)
    loss_miss = tf.reduce_sum(groups_miss)

    # Calculate the extra_miss
    k = tf.reduce_sum(errors_extra, axis=1)
    groups_extra = tf.math.sqrt(k)
    loss_extra = tf.reduce_sum(groups_extra)

    # Enter the loss for the sample
    return (loss_miss + loss_extra) / 2.0

def onlyFasterLasso(y,p):
    # Get the ones and zeros as ragged tensors
    bool_y = tf.cast(y, tf.bool)
    ones = tf.ragged.boolean_mask(p, bool_y)

    inverted_labels_bool = tf.math.logical_not(bool_y)
    zeros = tf.ragged.boolean_mask(p, inverted_labels_bool)

    # Stack corresponding ones and zeros on top of each other
    one_zero_pairs = tf.ragged.stack([ones,zeros], axis=1)

    CLASSES = []
    LOSSES = tf.map_fn(calc_errors, one_zero_pairs, fn_output_signature=tf.float32, parallel_iterations=32)

    return LOSSES, CLASSES

@tf.function
def onlyLasso(y,p):
    # Get the ones and zeros as ragged tensors
    bool_y = tf.cast(y, tf.bool)
    ones = tf.ragged.boolean_mask(p, bool_y)

    inverted_labels_bool = tf.math.logical_not(bool_y)
    zeros = tf.ragged.boolean_mask(p, inverted_labels_bool)

    # Stack corresponding ones and zeros on top of each other
    one_zero_pairs = tf.ragged.stack([ones,zeros], axis=1)

    indVar = 0
    LOSSES = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    CLASSES = []

    # Fix this if you can!
    for one_zero_pair in one_zero_pairs:
        # Calculate missing class errors
        onez = tf.expand_dims(one_zero_pair[0], axis=0)
        zeroz = tf.expand_dims(one_zero_pair[1], axis=1)
        k = tf.math.subtract(zeroz, onez)
        k = tf.math.multiply(k, 2)
        k = tf.math.add(k, 1)
        k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
        errors_miss = tf.math.square(k)

        # Calculate extra class errors
        onez = tf.expand_dims(one_zero_pair[0], axis=1)
        zeroz = tf.expand_dims(one_zero_pair[1], axis=0)
        k = tf.math.subtract(zeroz, onez)
        k = tf.math.multiply(k, 2)
        k = tf.math.add(k, 1)
        k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
        errors_extra = tf.math.square(k)

        # Calculate the loss_miss
        k = tf.reduce_sum(errors_miss, axis=1)
        groups_miss = tf.math.sqrt(k)
        loss_miss = tf.reduce_sum(groups_miss)

        # Calculate the extra_miss
        k = tf.reduce_sum(errors_extra, axis=1)
        groups_extra = tf.math.sqrt(k)
        loss_extra = tf.reduce_sum(groups_extra)

        # Enter the loss for the sample
        LOSSES = LOSSES.write(indVar, (loss_miss + loss_extra) / 2.0)

        indVar += 1

    return LOSSES.stack(), CLASSES

@tf.function
def fasterLasso(y,p):
    '''
    input: predictions from DNN as y. Its shape is (batch_size, num_of_classes)
           labels of the predicted samples. Its shape is (batch_size, num_of_classes)
    output: LOSSES tensor that has (batch_size) errors for each sample
            CLASSES tensor that has the index of the potential noisy class for each sample.

    Group Lasso Error for missing classes: E(zero_classes) sqrt( E(one_classes) F(K,L)^2 )
    Group Lasso Error for extra classes: E(one_classes) sqrt( E(zero_classes) F(K,L)^2 )
    E(x) is the cumulative sum!
    THE DISCRIMINATIVE FUNCTION IS F(K,L) = MAX(0, 2*(L-K)+1)
    ARTIFICIAL TRESHHOLD IS 0.5 FOR BOTH OF THEM
    For the true classification, the function gives -1
    For a missing class or an extra class, the function gives +1
    And for the extreme, which is missing class and extra class together, the function gives +3

    How to figure if Missing class or Extra class?
    Look at the respective error group if all the elements of the subarray are non-zero, this
    element is missing class or extra class respectively. However, this method does not give
    always the right answers. So if there are more than one missing classes or extra labels just
    choose one with the biggest loss.

    YOU CANT MEASURE HOW GOOD YOU ARE DOING COMPARED TO THE TRUE LABEL OF A SAMPLE!
    '''
    # Get the ones and zeros as ragged tensors. For each sample the number of ones and zeros are different. That's why it is a ragged tensor
    bool_y = tf.cast(y, tf.bool) # Transform the labels into a boolean tensor to get positive classes
    ones = tf.ragged.boolean_mask(p, bool_y)

    inverted_labels_bool = tf.math.logical_not(bool_y)
    bool_y = tf.cast(y, tf.bool) # Transform the labels into a boolean tensor to get negative  classes
    zeros = tf.ragged.boolean_mask(p, inverted_labels_bool)

    # This includes positive and negative class pairs for every sample in the batch
    one_zero_pairs = tf.ragged.stack([ones,zeros], axis=1) # Stack corresponding ones and zeros on top of each other

    indVar = 0 # Increment this value to write loss and class values into the corresponding tensor arrays
    LOSSES = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    CLASSES = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    # Fix this if you can!
    for one_zero_pair in one_zero_pairs:
        # Calculate missing class errors
        onez = tf.expand_dims(one_zero_pair[0], axis=0) # Expand dims to do pairwise calculations
        zeroz = tf.expand_dims(one_zero_pair[1], axis=1)
        k = tf.math.subtract(zeroz, onez)
        k = tf.math.multiply(k, 2)
        k = tf.math.add(k, 1)
        k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
        errors_miss = tf.math.square(k)
        # Calculate extra class errors
        onez = tf.expand_dims(one_zero_pair[0], axis=1) # Expand dims to do pairwise calculations
        zeroz = tf.expand_dims(one_zero_pair[1], axis=0)
        k = tf.math.subtract(zeroz, onez)
        k = tf.math.multiply(k, 2)
        k = tf.math.add(k, 1)
        k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
        errors_extra = tf.math.square(k)

        # Calculate the loss_miss
        k = tf.reduce_sum(errors_miss, axis=1)
        groups_miss = tf.math.sqrt(k)
        loss_miss = tf.reduce_sum(groups_miss)
        # Calculate the extra_miss
        k = tf.reduce_sum(errors_extra, axis=1)
        groups_extra = tf.math.sqrt(k)
        loss_extra = tf.reduce_sum(groups_extra)

        # Find the noisy missing class, choose the one with the biggest loss
        # Index that I get is not the real index of the missing class. It is the index of it in its own group, which is zeros!
        # Find the noisy extra class, choose the one with the biggest loss
        # Index that I get is not the real index of the missing class. It is the index of it in its own group, which is zeros!
        onez_inds = tf.where(bool_y[0])
        zeroz_inds = tf.where(tf.math.logical_not(bool_y[0]))

        row_wise_min_miss = tf.reduce_min(errors_miss, axis=1) # Get the min element in the errors for classes to figure if all the errors are non-zero
        row_wise_min_extra = tf.reduce_min(errors_extra, axis=1)
        indices_miss = tf.where(tf.math.not_equal(row_wise_min_miss, 0)) 
        indices_extra = tf.where(tf.math.not_equal(row_wise_min_extra, 0))
        potential_missing_classes = tf.gather(groups_miss, indices_miss)
        potential_extra_classes = tf.gather(groups_extra, indices_extra)

        try:
            potential_missing = tf.math.argmax(potential_missing_classes)
            missing_class_index = tf.gather(indices_miss, potential_missing)
        except:
            missing_class_index = None
        try:
            potential_extra = tf.math.argmax(potential_extra_classes)
            extra_class_index = tf.gather(indices_extra, potential_extra)
        except:
            extra_class_index = None

        # Enter the loss for the sample
        LOSSES = LOSSES.write(indVar, (loss_miss + loss_extra) / 2.0)
        CLASSES = CLASSES.write(indVar, missing_class_index)

        indVar += 1

    return LOSSES.stack(), CLASSES.stack()

def bucakLasso(y,p):

    groups_miss = []
    errors_miss = []
    loss_miss = 0
    missing_classes = set()

    for i, l in np.ndenumerate(p):
        if y[i] == -1:
            group = 0
            errors = []
            for idx, k in np.ndenumerate(p):
                if y[idx] == 1:
                    error = (max(0, (l-k)+1) ** 2)
                    group += error
                    errors.append(error)
            groups_miss.append(math.sqrt(group))
            missing_classes.add(i[0])
            errors_miss.append(errors)
    
    for i in groups_miss:
        loss_miss += i

    return loss_miss, groups_miss, errors_miss, missing_classes

def calc_missing(p, y):
    groups_miss = []
    errors_miss = []
    loss_miss = 0
    missing_classes = []

    for i, l in enumerate(p):
        if y[i] == 0:
            group = 0
            errors = []
            for idx, k in enumerate(p):
                if y[idx] == 1:
                    error = (max(0, 2*(l-k)+1) ** 2)
                    group += error
                    errors.append(error)

            # This is just to find the missing labels
            if all(e != 0 for e in errors):
                missing_classes.append([math.sqrt(group), i])
            
            groups_miss.append(math.sqrt(group))
            errors_miss.append(errors)
    
    for i in groups_miss:
        loss_miss += i
    
    # Choose the one with biggest loss
    try:
        missing_class = max(missing_classes)
    except:
        missing_class = None

    return groups_miss, errors_miss, loss_miss, missing_classes, missing_class

def calc_extra(p, y):
    groups_extra = []
    errors_extra = []
    loss_extra = 0
    extra_classes = []
    
    for i, k in enumerate(p):
        if y[i] == 1:
            group = 0
            errors = []
            for idx, l in enumerate(p):
                if y[idx] == 0:
                    error = (max(0, 2*(l-k)+1) ** 2)
                    errors.append(error)
                    group += error
            
            # This is just to find the extra labels
            if all(e != 0 for e in errors):
                extra_classes.append([math.sqrt(group), i])
            
            groups_extra.append(math.sqrt(group))
            errors_extra.append(errors)

    for i in groups_extra:
        loss_extra += i

    # Choose the one with biggest loss
    try:
        extra_class = max(extra_classes)
    except:
        extra_class = None

    return groups_extra, errors_extra, loss_extra, extra_classes, extra_class

def oldLasso(ys,ps):
    
    '''
    THE DISCRIMINATIVE FUNCTION IS F(K,L) = MAX(0, 2*(L-K)+1)
    ARTIFICIAL TRESHHOLD IS 0.5 FOR BOTH OF THEM
    For the true classification, the function gives -1
    For a missing class or an extra class, the function gives +1
    And for the extreme, which is missing class and extra class together, the function gives +3
    
    How to figure if Missing class or Extra class?
    Look at the respective error group if all the elements of the subarray are non-zero, this
    element is missing class or extra class respectively. However, this method does not give
    always the right answers. So if there are more than one missing classes or extra labels just
    choose one with the biggest loss.
    
    YOU CANT MEASURE HOW GOOD YOU ARE DOING COMPARED TO THE TRUE LABEL OF A SAMPLE! 
    '''
    
    LOSSES = []
    CLASSES = []
    results = []

    for y, p in zip(ys,ps):

        # FIND THE MISSING CLASS LABELS
        groups_miss, errors_miss, loss_miss, missing_classes, missing_class = calc_missing(p, y)        
        print(f"errors_miss: {errors_miss}")
        
        # TO FIND THE EXTRA CLASS LABELS
        groups_extra, errors_extra, loss_extra, extra_classes, extra_class = calc_extra(p,y)

        LOSS = (loss_miss + loss_extra) / 2.0

        if missing_class == None and extra_class == None:
            change_class = None
        elif missing_class == None and extra_class != None:
            change_class = extra_class[1]
        elif missing_class != None and extra_class == None:
            change_class = missing_class[1]
        elif missing_class[0] > extra_class[0]:
            change_class = missing_class[1]
        else:
            change_class = extra_class[1]

        LOSSES.append(LOSS)
        CLASSES.append(change_class)

        results.append([LOSS, change_class, missing_class, extra_class, loss_miss, groups_miss, errors_miss, loss_extra, groups_extra, errors_extra, missing_classes, extra_classes])
    
    return np.array(LOSSES), np.array(CLASSES)


def test():
    y1 = np.array([[1,0,1,0,1,1,0,0,1]]) # 9 CLASSES
    #noisy_y1 = np.array([[1,0,0,0,1,0,0,0,1]]) # Missing classes at indexes 2 and 5
    #noisy_y1 = np.array([[1,0,1,1,1,1,0,1,1]]) # Extra classes at indexes 3 and 7
    noisy_y1 = np.array([[1,0,0,0,1,1,0,1,1]]) # Missing class at index 2 and extra class at index 7
    double_noisy_y1 = np.array([[1,0,0,0,1,0,0,1,1],[1,0,0,0,1,1,0,1,1]]) # Missing class at index 2 and extra class at index 7
    p1 = np.array([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]])
    p2 = np.array([[0.78,0.34,0.72,0.2,0.86,0.62,0.4,0.17,0.95]])
    p3 = np.array([[0.53,0.47,0.58,0.39,0.51,0.54,0.49,0.46,0.61]])
    p4 = np.array([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.78,0.34,0.72,0.2,0.86,0.62,0.4,0.17,0.95]], dtype=np.float32)
    losses, classes = newLasso(double_noisy_y1,p4)
    losses1, classes1 = groupLasso(double_noisy_y1,p4)

    print(f"losses: {losses}")
    print(f"classes: {classes}")
    print(f"losses1: {losses1}")

    '''
    results = groupLasso(double_noisy_y1,p4)
    print(f"LOSS1: {results[0][0]}")
    print(f"change_class: {results[0][1]}")
    print(f"missing_class: {results[0][2]}")
    print(f"extra_class: {results[0][3]}")
    print(f"loss_miss: {results[0][4]}")
    print(f"groups_miss: {results[0][5]}")
    print(f"errors_miss: {results[0][6]}")
    print(f"loss_extra: {results[0][7]}")
    print(f"groups_extra: {results[0][8]}")
    print(f"errors_extra: {results[0][9]}")
    print(f"missing_classes: {results[0][10]}")
    print(f"extra_classes: {results[0][11]}")
    '''

def bucakTest():
    y1 = np.array([1,-1,1,-1,1,1,-1,-1,1]) # 9 CLASSES
    noisy_y1 = np.array([1,-1,-1,-1,1,-1,-1,-1,1]) # Missing classes at indexes 2 and 5
    #noisy_y1 = np.array([1,-1,1,1,1,1,-1,1,1]) # Extra classes at indexes 3 and 7
    #noisy_y1 = np.array([1,-1,-1,-1,1,1,-1,1,1]) # Missing class at index 2 and extra class at index 7
    p1 = np.array([0.91,-0.88,0.87,-0.83,0.92,0.83,-0.93,-0.91,0.95])  # loss = 3.1990626168794223
    p2 = np.array([0.78,-0.68,0.72,-0.8,0.86,0.62,-0.32,-0.56,0.95])   # loss = 2.804749814873396
    p3 = np.array([0.23,-0.32,0.42,-0.52,0.27,0.37,-0.32,-0.13,0.19])   # loss = 7.14685765636982
    p4 = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])   # loss = 10.392304845413262 
    p5 = np.array([1.,-1.,1.,-1.,1.,1.,-1.,-1.,1.])   # loss = 3.4641016151377544
    p6 = np.array([1.,-1.,-1.,-1.,1.,-1.,-1.,-1.,1.])   # loss = 0.0
    p7 = np.array([.02,-.02,.02,-.02,.02,.02,-.02,-.02,.02])
    p8 = np.array([.4,-.4,.4,-.4,.4,.4,-.4,-.4,.4])
    loss_miss1, groups_miss1, errors_miss1, missing_classes1 = bucakLasso(y1,p8)
    print(f"loss_miss1: {loss_miss1}")
    print(f"groups_miss1: {groups_miss1}")
    print(f"errors_miss1: {errors_miss1}")
    print(f"missing_classes1: {missing_classes1}")

# test()
bucakTest()
