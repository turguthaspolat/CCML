'''
File: printers.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports 
import tensorflow as tf

def train_printer(epoch, L2, L3, trainMetrics):
    print(f'------- EPOCH {epoch} TRAINING -------')
    print(f'L2 to be maximized over epoch {epoch}: {L2}')
    print(f'L3 to be minimized over epoch {epoch}: {L3}')
    print(f'Training Accuracy 1 over epoch {epoch}: {trainMetrics.acc1.result():.3f}')
    print(f'Training Accuracy 2 over epoch {epoch}: {trainMetrics.acc2.result():.3f}')
    print(f'Training Precision 1 over epoch {epoch}: {trainMetrics.precision1.result():.3f}')
    print(f'Training Precision 2 over epoch {epoch}: {trainMetrics.precision2.result():.3f}')
    print(f'Training Recall 1 over epoch {epoch}: {trainMetrics.recall1.result():.3f}')
    print(f'Training Recall 2 over epoch {epoch}: {trainMetrics.recall2.result():.3f}')
    print(f'-----------------------------')

def val_printer(epoch, valMetrics):
    print(f'------- EPOCH {epoch} VALIDATION -------')
    print(f'Validation Accuracy 1 over epoch {epoch}: {valMetrics.acc1.result():.3f}')
    print(f'Validation Accuracy 2 over epoch {epoch}: {valMetrics.acc2.result():.3f}')
    print(f'Validation Precision 1 over epoch {epoch}: {valMetrics.precision1.result():.3f}')
    print(f'Validation Precision 2 over epoch {epoch}: {valMetrics.precision2.result():.3f}')
    print(f'Validation Recall 1 over epoch {epoch}: {valMetrics.recall1.result():.3f}')
    print(f'Validation Recall 2 over epoch {epoch}: {valMetrics.recall2.result():.3f}')
    try:
        f1_1 = 2.0 * (valMetrics.precision1.result() * valMetrics.recall1.result()) / \
            (valMetrics.precision1.result() + valMetrics.recall1.result())
    except:
        f1_1 = 0.0
    try:
        f1_2 = 2.0 * (valMetrics.precision2.result() * valMetrics.recall2.result()) / \
            (valMetrics.precision2.result() + valMetrics.recall2.result())
    except:
        f1_2 = 0.0
    print(f'Validation F1 1 over epoch {epoch}: {f1_1:.3f}')
    print(f'Validation F1 2 over epoch {epoch}: {f1_2:.3f}')
    # print(f'Confusion matric for model 1 over epoch {epoch}: {valMetrics.confmat1.result()}')
    # print(f'Confusion matric for model 2 over epoch {epoch}: {valMetrics.confmat2.result()}')

    classwise_precision_1 = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[0,1]), valMetrics.confmat1.result())
    print(f'Classwise Precision 1 over epoch {epoch}: {classwise_precision_1}')
    classwise_precision_2 = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[0,1]), valMetrics.confmat2.result())
    print(f'Classwise Precision 2 over epoch {epoch}: {classwise_precision_2}')
    classwise_recall_1 = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[1,0]), valMetrics.confmat1.result())
    print(f'Classwise Recall 1 over epoch {epoch}: {classwise_recall_1}')
    classwise_recall_2 = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[1,0]), valMetrics.confmat2.result())
    print(f'Classwise Recall 2 over epoch {epoch}: {classwise_recall_2}')
    try:
        classwise_f1_1 = (2.0 * (classwise_recall_1 * classwise_precision_1)) / \
            (classwise_recall_1 + classwise_precision_1)
    except:
        classwise_f1_1 = 0.0
    print(f'Classwise F1 1 over epoch {epoch}: {classwise_f1_1}')
    try:
        classwise_f1_2 = (2.0 * (classwise_recall_2 * classwise_precision_2)) / \
            (classwise_recall_2 + classwise_precision_2)
    except:
        classwise_f1_2 = 0.0
    print(f'Classwise F1 2 over epoch {epoch}: {classwise_f1_2}')
    classwise_positive_labels = tf.map_fn(lambda t: t[1,0] + t[1,1], valMetrics.confmat1.result())
    print(f'Classwise Positive Labels over epoch {epoch}: {classwise_positive_labels}')
    classwise_negative_labels = tf.map_fn(lambda t: t[0,1] + t[0,0], valMetrics.confmat1.result())
    print(f'Classwise Negative Labels over epoch {epoch}: {classwise_negative_labels}')
    print(f'-----------------------------')

def test_printer(testMetrics):
    print(f'------- TEST RESULTS -------')
    print(f'Test Accuracy 1: {testMetrics.acc1.result():.3f}')
    print(f'Test Accuracy 2: {testMetrics.acc2.result():.3f}')
    print(f'Test Precision 1: {testMetrics.precision1.result():.3f}')
    print(f'Test Precision 2: {testMetrics.precision2.result():.3f}')
    print(f'Test Recall 1: {testMetrics.recall1.result():.3f}')
    print(f'Test Recall 2: {testMetrics.recall2.result():.3f}')
    print(f'Test F1 None 1: {testMetrics.f1none_1.result()}')
    print(f'Test F2 None 2: {testMetrics.f1none_2.result()}')
    print(f'Test F1 Micro 1: {testMetrics.f1micro_1.result():.3f}')
    print(f'Test F2 Micro 2: {testMetrics.f1micro_2.result():.3f}')
    print(f'Test F1 Macro 1: {testMetrics.f1macro_1.result():.3f}')
    print(f'Test F2 Macro 2: {testMetrics.f1macro_2.result():.3f}')
    print(f'Test F1 Weighted 1: {testMetrics.f1weighted_1.result():.3f}')
    print(f'Test F2 Weighted 2: {testMetrics.f1weighted_2.result():.3f}')
    try:
        f1_1 = 2.0 * (testMetrics.precision1.result() * testMetrics.recall1.result()) / \
            (testMetrics.precision1.result() + testMetrics.recall1.result())
    except:
        f1_1 = 0.0
    try:
        f1_2 = 2.0 * (testMetrics.precision2.result() * testMetrics.recall2.result()) / \
            (testMetrics.precision2.result() + testMetrics.recall2.result())
    except:
        f1_2 = 0.0
    print(f'Test F1 1: {f1_1:.3f}')
    print(f'Test F1 2: {f1_2:.3f}')
    # print(f"Confusion matric for model 1: {testMetrics.confmat1.result()}")
    # print(f"Confusion matric for model 2: {testMetrics.confmat2.result()}")

    classwise_precision_1 = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[0,1]), testMetrics.confmat1.result())
    print(f'Classwise Precision 1: {classwise_precision_1}')
    classwise_precision_2 = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[0,1]), testMetrics.confmat2.result())
    print(f'Classwise Precision 2: {classwise_precision_2}')
    classwise_recall_1 = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[1,0]), testMetrics.confmat1.result())
    print(f'Classwise Recall 1: {classwise_recall_1}')
    classwise_recall_2 = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[1,0]), testMetrics.confmat2.result())
    print(f'Classwise Recall 2: {classwise_recall_2}')
    try:
        classwise_f1_1 = (2.0 * (classwise_recall_1 * classwise_precision_1)) / \
            (classwise_recall_1 + classwise_precision_1)
    except:
        classwise_f1_1 = 0.0
    print(f'Classwise F1 1: {classwise_f1_1}')
    try:
        classwise_f1_2 = (2.0 * (classwise_recall_2 * classwise_precision_2)) / \
            (classwise_recall_2 + classwise_precision_2)
    except:
        classwise_f1_2 = 0.0
    print(f'Classwise F1 2: {classwise_f1_2}')
    classwise_positive_labels = tf.map_fn(lambda t: t[1,0] + t[1,1], testMetrics.confmat1.result())
    print(f'Classwise Positive Labels: {classwise_positive_labels}')
    classwise_negative_labels = tf.map_fn(lambda t: t[0,1] + t[0,0], testMetrics.confmat1.result())
    print(f'Classwise Negative Labels: {classwise_negative_labels}')
    print(f'-----------------------------')
