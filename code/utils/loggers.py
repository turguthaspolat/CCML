'''
File: loggers.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de) 
'''

# Third party imports 
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

class TrainMetrics:
    def __init__(self, logdir, num_classes):
        self.acc1 = keras.metrics.Accuracy('train_acc_for_model_1')
        self.loss1 = keras.metrics.Mean('train_loss_for_model_1', dtype=tf.float32)
        self.acc2 = keras.metrics.Accuracy('train_acc_for_model_2')
        self.loss2 = keras.metrics.Mean('train_loss_for_model_2', dtype=tf.float32)
        self.precision1 = keras.metrics.Precision(name='train_precision_1')
        self.precision2 = keras.metrics.Precision(name='train_precision_2')
        self.recall1 = keras.metrics.Recall(name='train_recall_1')
        self.recall2 = keras.metrics.Recall(name='train_recall_2')
        self.confmat1 = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes, name='confusion_matrix_1')
        self.confmat2 = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes, name='confusion_matrix_2')
        self.summary1 = tf.summary.create_file_writer(logdir + '/model1/train')
        self.summary2 = tf.summary.create_file_writer(logdir + '/model2/train')
    
    def reset_states(self):
        self.acc1.reset_states()
        self.acc2.reset_states()
        self.loss1.reset_states()
        self.loss2.reset_states()
        self.precision1.reset_states()
        self.precision2.reset_states()
        self.recall1.reset_states()
        self.recall2.reset_states()
        self.confmat1.reset_states()
        self.confmat2.reset_states()
    
    def update_states(self, y_batch, loss_value1, loss_value2, logits1, logits2):
        self.loss1.update_state(loss_value1)
        self.loss2.update_state(loss_value2)
        self.acc1.update_state(y_batch, logits1)
        self.acc2.update_state(y_batch, logits2)
        self.precision1.update_state(y_batch, logits1)
        self.precision2.update_state(y_batch, logits2)
        self.recall1.update_state(y_batch, logits1)
        self.recall2.update_state(y_batch, logits2)
        self.confmat1.update_state(y_batch, logits1)
        self.confmat2.update_state(y_batch, logits2)
    
    def write_summary(self, epoch, L2, L3):
        with self.summary1.as_default():
            tf.summary.scalar('Loss', self.loss1.result(), step=epoch)
            tf.summary.scalar('Accuracy', self.acc1.result(), step=epoch)
            tf.summary.scalar('L2', float(L2), step=epoch)
            tf.summary.scalar('L3', float(L3), step=epoch)
            tf.summary.scalar('Precision', self.precision1.result(), step=epoch)
            tf.summary.scalar('Recall', self.recall1.result(), step=epoch)
            f1_1 = 2.0 * (self.precision1.result() * self.recall1.result()) / \
                (self.precision1.result() + self.recall1.result())
            tf.summary.scalar('F1', f1_1, step=epoch)
        with self.summary2.as_default():
            tf.summary.scalar('Loss', self.loss2.result(), step=epoch)
            tf.summary.scalar('Accuracy', self.acc2.result(), step=epoch)
            tf.summary.scalar('L2', float(L2), step=epoch)
            tf.summary.scalar('L3', float(L3), step=epoch)
            tf.summary.scalar('Precision', self.precision2.result(), step=epoch)
            tf.summary.scalar('Recall', self.recall2.result(), step=epoch)
            f1_2 = 2.0 * (self.precision2.result() * self.recall2.result()) / \
                (self.precision2.result() + self.recall2.result())
            tf.summary.scalar('F1', f1_2, step=epoch)

class ValMetrics:
    def __init__(self, logdir, num_classes):
        self.acc1 = keras.metrics.Accuracy('val_acc_for_model_1')
        self.acc2 = keras.metrics.Accuracy('val_acc_for_model_2')
        self.precision1 = keras.metrics.Precision(name='val_precision_1')
        self.precision2 = keras.metrics.Precision(name='val_precision_2')
        self.recall1 = keras.metrics.Recall(name='val_recall_1')
        self.recall2 = keras.metrics.Recall(name='val_recall_2')
        self.confmat1 = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes, name='confusion_matrix_1')
        self.confmat2 = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes, name='confusion_matrix_2')
        self.summary1 = tf.summary.create_file_writer(logdir + '/model1/val')
        self.summary2 = tf.summary.create_file_writer(logdir + '/model2/val')
        self.best_f1 = 0
    
    def reset_states(self):
        self.acc1.reset_states()
        self.acc2.reset_states()
        self.precision1.reset_states()
        self.precision2.reset_states()
        self.recall1.reset_states()
        self.recall2.reset_states()
        self.confmat1.reset_states()
        self.confmat2.reset_states()
    
    def update_states(self, y_batch, logits1, logits2):
        self.acc1.update_state(y_batch, logits1)
        self.acc2.update_state(y_batch, logits2)
        self.precision1.update_state(y_batch, logits1)
        self.precision2.update_state(y_batch, logits2)
        self.recall1.update_state(y_batch, logits1)
        self.recall2.update_state(y_batch, logits2)
        self.confmat1.update_state(y_batch, logits1)
        self.confmat2.update_state(y_batch, logits2)
    
    def write_summary(self, epoch):
        with self.summary1.as_default():
            tf.summary.scalar('Accuracy', self.acc1.result(), step=epoch)
            tf.summary.scalar('Precision', self.precision1.result(), step=epoch)
            tf.summary.scalar('Recall', self.recall1.result(), step=epoch)
            try:
                f1_1 = 2.0 * (self.precision1.result() * self.recall1.result()) / \
                    (self.precision1.result() + self.recall1.result())
            except:
                f1_1 = 0.0
            tf.summary.scalar('F1', f1_1, step=epoch)
        with self.summary2.as_default():
            tf.summary.scalar('Accuracy', self.acc2.result(), step=epoch)
            tf.summary.scalar('Precision', self.precision2.result(), step=epoch)
            tf.summary.scalar('Recall', self.recall2.result(), step=epoch)
            try:
                f1_2 = 2.0 * (self.precision2.result() * self.recall2.result()) / \
                    (self.precision2.result() + self.recall2.result())
            except:
                f1_2 = 0.0
            tf.summary.scalar('F1', f1_2, step=epoch)
    
    def save_best_model(self, model1, model2, noise_rate, epoch):
        f1_1 = 2.0 * (self.precision1.result() * self.recall1.result()) / \
                    (self.precision1.result() + self.recall1.result())
        f1_2 = 2.0 * (self.precision2.result() * self.recall2.result()) / \
                    (self.precision2.result() + self.recall2.result())
        
        # Choose the better f1 score among two
        new_f1 = f1_1 if f1_1 > f1_2 else f1_2
        
        # Choose the better f1 score's model
        new_model = model1 if f1_1 > f1_2 else model2
        
        if new_f1 > self.best_f1:
            # Save the model
            new_model.saveModel('best', noise_rate)
            print(f'New best model at epoch {epoch} with an f1 score of {new_f1}')
            
            # Update the best f1
            self.best_f1 = new_f1
    
class TestMetrics:
    def __init__(self, logdir, num_classes):
        self.acc1 = keras.metrics.Accuracy('test_acc_for_model_1')
        self.acc2 = keras.metrics.Accuracy('test_acc_for_model_2')
        self.precision1 = keras.metrics.Precision(name='test_precision_1')
        self.precision2 = keras.metrics.Precision(name='test_precision_2')
        self.recall1 = keras.metrics.Recall(name='test_recall_1')
        self.recall2 = keras.metrics.Recall(name='test_recall_2')
        self.f1none_1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5)
        self.f1none_2 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5)
        self.f1micro_1 = tfa.metrics.F1Score(num_classes=num_classes, average='micro', threshold=0.5)
        self.f1micro_2 = tfa.metrics.F1Score(num_classes=num_classes, average='micro', threshold=0.5)
        self.f1macro_1 = tfa.metrics.F1Score(num_classes=num_classes, average='macro', threshold=0.5)
        self.f1macro_2 = tfa.metrics.F1Score(num_classes=num_classes, average='macro', threshold=0.5)
        self.f1weighted_1 = tfa.metrics.F1Score(num_classes=num_classes, average='weighted', threshold=0.5)
        self.f1weighted_2 = tfa.metrics.F1Score(num_classes=num_classes, average='weighted', threshold=0.5)
        self.confmat1 = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes, name='confusion_matrix_1')
        self.confmat2 = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes, name='confusion_matrix_2')
        self.summary1 = tf.summary.create_file_writer(logdir + '/model1/test')
        self.summary2 = tf.summary.create_file_writer(logdir + '/model2/test')
    
    def reset_states(self):
        self.acc1.reset_states()
        self.acc2.reset_states()
        self.precision1.reset_states()
        self.precision2.reset_states()
        self.recall1.reset_states()
        self.recall2.reset_states()
        self.f1none_1.reset_states()
        self.f1none_2.reset_states()
        self.f1micro_1.reset_states()
        self.f1micro_2.reset_states()
        self.f1macro_1.reset_states()
        self.f1macro_2.reset_states()
        self.f1weighted_1.reset_states()
        self.f1weighted_2.reset_states()
        self.confmat1.reset_states()
        self.confmat2.reset_states()
    
    def update_states(self, y_batch, logits1, logits2):
        self.acc1.update_state(y_batch, logits1)
        self.acc2.update_state(y_batch, logits2)
        self.precision1.update_state(y_batch, logits1)
        self.precision2.update_state(y_batch, logits2)
        self.recall1.update_state(y_batch, logits1)
        self.recall2.update_state(y_batch, logits2)
        self.f1none_1.update_state(y_batch, logits1)
        self.f1none_2.update_state(y_batch, logits2)
        self.f1micro_1.update_state(y_batch, logits1)
        self.f1micro_2.update_state(y_batch, logits2)
        self.f1macro_1.update_state(y_batch, logits1)
        self.f1macro_2.update_state(y_batch, logits2)
        self.f1weighted_1.update_state(y_batch, logits1)
        self.f1weighted_2.update_state(y_batch, logits2)
        self.confmat1.update_state(y_batch, logits1)
        self.confmat2.update_state(y_batch, logits2)
    '''
    def write_summary(self):
        with self.summary1.as_default():
            tf.summary.scalar('Accuracy', self.acc1.result())
            tf.summary.scalar('Precision', self.precision1.result())
            tf.summary.scalar('Recall', self.recall1.result())
            try:
                f1_1 = 2.0 * (self.precision1.result() * self.recall1.result()) / \
                    (self.precision1.result() + self.recall1.result())
            except:
                f1_1 = 0.0
            tf.summary.scalar('F1', f1_1)
        with self.summary2.as_default():
            tf.summary.scalar('Accuracy', self.acc2.result())
            tf.summary.scalar('Precision', self.precision2.result())
            tf.summary.scalar('Recall', self.recall2.result())
            try:
                f1_2 = 2.0 * (self.precision2.result() * self.recall2.result()) / \
                    (self.precision2.result() + self.recall2.result())
            except:
                f1_2 = 0.0
            tf.summary.scalar('F1', f1_2)
    '''