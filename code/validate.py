'''
File: validate.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports
import tensorflow as tf

# Local imports
from data_prep.stackBands import prepare_input
from utils.printers import val_printer

def validate(args, valMetrics, pred_treshold, epoch, model1, model2):
    
    for batch in args.val_data:
        if args.label_type == 'ucmerced':
            x_batch_val = batch[0]
            x_batch_val = tf.cast(x_batch_val, tf.float32)
            y_batch_val = batch[1]
            y_batch_val = tf.cast(y_batch_val, tf.float32)
        else:
            if args.channels == 'RGB':
                x_batch_val = tf.stack([batch[0]['B04'], batch[0]['B03'], batch[0]['B02']], axis=3)
            else:
                x_batch_val = prepare_input(batch[0])
            y_batch_val = batch[1]['labels']
        
        val_logits_1, _ = args.model1(x_batch_val, training=False)
        val_logits_2, _ = args.model2(x_batch_val, training=False)

        accuracy_val_logits_1 = tf.cast(tf.math.sigmoid(val_logits_1) >= pred_treshold, tf.float32)
        accuracy_val_logits_2 = tf.cast(tf.math.sigmoid(val_logits_2) >= pred_treshold, tf.float32)

        valMetrics.update_states(y_batch_val, accuracy_val_logits_1, accuracy_val_logits_2)
    
    val_printer(epoch, valMetrics)
        
    valMetrics.write_summary(epoch)

    valMetrics.save_best_model(model1, model2, args.sample_rate, epoch)
     
    valMetrics.reset_states()
