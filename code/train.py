'''
File: train.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports
import tensorflow as tf

# Local imports
from loss_fun import loss_fun
from utils.printers import train_printer
from data_prep.stackBands import prepare_input

def train(args, noisifier, pred_treshold, optimizer_1, optimizer_2, trainMetrics, epoch, eximage, start_flipping):

    print(f'------- EPOCH {epoch} TRAINING LOSSES -------')

    # Variables for statistics
    num_positive_training_labels = 0
    num_negative_training_labels = 0
    
    for step, batch in enumerate(args.train_data):
        if args.label_type == 'ucmerced':
            # UC Merced is RGB default.
            x_batch_train = batch[0]
            x_batch_train = tf.cast(x_batch_train, tf.float32)
            y_batch_train = batch[1]
            y_batch_train = tf.cast(y_batch_train, tf.float32)
        else:
            if args.channels == 'RGB':
                # Use only RGB bands
                x_batch_train = tf.stack([batch[0]['B04'], batch[0]['B03'], batch[0]['B02']], axis=3)
            else:
                x_batch_train = prepare_input(batch[0])
            y_batch_train = batch[1]['labels']
        
        accuracy_logits_1, accuracy_logits_2, logits_1, l2_logits_m1, logits_2, l2_logits_m2, \
        loss_value_1, loss_value_2, L3, L2 = train_loop(pred_treshold, optimizer_1, \
        optimizer_2, trainMetrics, epoch, x_batch_train, y_batch_train, batch, eximage, noisifier, start_flipping, args)

        trainMetrics.update_states(y_batch_train, loss_value_1, loss_value_2, accuracy_logits_1, accuracy_logits_2)
        print(f'Batch {step+1} === Training Loss 1: {loss_value_1}. Training Loss 2: {loss_value_2}')
        
        num_positive_training_labels += tf.map_fn(lambda t: t[1,0] + t[1,1], trainMetrics.confmat1.result())
        num_negative_training_labels += tf.map_fn(lambda t: t[0,1] + t[0,0], trainMetrics.confmat1.result())
        
    print(f'num_positive_training_labels: {num_positive_training_labels}')
    print(f'num_negative_training_labels: {num_negative_training_labels}')
    
    print(f'-----------------------------')
    train_printer(epoch, L2, L3, trainMetrics)
    
    trainMetrics.write_summary(epoch, L2, L3)
    
    trainMetrics.reset_states()
    

def train_loop(pred_treshold, optimizer_1, optimizer_2, trainMetrics, epoch, \
        x_batch_train, y_batch_train, batch, eximage, noisifier, start_flipping, args):    
    
    true_y_batch = y_batch_train
    # Add random noise
    # if add_noise and start_flipping > flip_bound:
    if args.add_noise:
        if args.noise_type == 1:
            y_batch_train = noisifier.random_multi_label_noise(y_batch_train, args.sample_rate, args.class_rate, seed=True)
        elif args.noise_type == 2:
            y_batch_train = noisifier.add_missing_extra_noise(y_batch_train, args.sample_rate, seed=True)
        y_batch_train = tf.convert_to_tensor(y_batch_train)

    with tf.GradientTape(persistent=True) as tape:
        
        logits_1, l2_logits_m1 = args.model1(x_batch_train, training=True)
        logits_2, l2_logits_m2 = args.model2(x_batch_train, training=True)
        loss_value_1, loss_value_2, L3, L2 = loss_fun(y_batch_train, logits_1, logits_2, 
                args.batch_size, l2_logits_m1, l2_logits_m2, args.sigma, args.swap, args.swap_rate, 
                args.lambda2, args.lambda3, args.flip_bound, start_flipping, args.flip_per, 
                args.miss_alpha, args.extra_beta, args.divergence_metric, args.alpha, pred_treshold)

    grads_1 = tape.gradient(loss_value_1, args.model1.trainable_weights)
    grads_2 = tape.gradient(loss_value_2, args.model2.trainable_weights)
    
    optimizer_1.apply_gradients(zip(grads_1, args.model1.trainable_weights))
    optimizer_2.apply_gradients(zip(grads_2, args.model2.trainable_weights))

    accuracy_logits_1 = tf.cast(tf.math.sigmoid(logits_1) >= pred_treshold, tf.float32)
    accuracy_logits_2 = tf.cast(tf.math.sigmoid(logits_2) >= pred_treshold, tf.float32)

    return accuracy_logits_1, accuracy_logits_2, logits_1, l2_logits_m1, logits_2, \
            l2_logits_m2, loss_value_1, loss_value_2, L3, L2
