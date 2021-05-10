'''
File: evaluate.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from data_prep.stackBands import prepare_input
from utils.printers import test_printer

def evaluate(pred_treshold, test_data, model1, model2, testMetrics, label_type, channels):

    for batch in test_data:
        
        if label_type == 'ucmerced':
            x_batch_test = batch[0]
            x_batch_test = tf.cast(x_batch_test, tf.float32)
            y_batch_test = batch[1]
            y_batch_test = tf.cast(y_batch_test, tf.float32)
        else:
            if channels == 'RGB':
                x_batch_test = tf.stack([batch[0]['B04'], batch[0]['B03'], batch[0]['B02']], axis=3)
            else:
                x_batch_test = prepare_input(batch[0])
            y_batch_test = batch[1]['labels']
        
        test_logits_1, _ = model1(x_batch_test, training=False)
        test_logits_2, _ = model2(x_batch_test, training=False)

        accuracy_test_logits_1 = tf.cast(tf.math.sigmoid(test_logits_1) >= pred_treshold, tf.float32)
        accuracy_test_logits_2 = tf.cast(tf.math.sigmoid(test_logits_2) >= pred_treshold, tf.float32)

        testMetrics.update_states(y_batch_test, accuracy_test_logits_1, accuracy_test_logits_2)
    
    test_printer(testMetrics)
    
    # testMetrics.write_summary()
    
    testMetrics.reset_states()
