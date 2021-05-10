import numpy as np
import tensorflow as tf
from groupLasso import groupLasso, choose_noisy_classes, flip_classes

'''
TESTS
'''
def start_endT(n,alpha,beta):
    noisy_y = tf.constant([[1,0,0,0,1,1,0,1,1],[1,0,0,0,1,1,0,1,1]]) # Missing class at index 2 and extra class at index 7
    p = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.78,0.34,0.72,0.2,0.86,0.62,0.4,0.17,0.95]], dtype=np.float32)
    losses, classes = groupLasso(noisy_y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def pot_class_assT(n,alpha,beta):
    noisy_y = tf.constant([[1,0,0,0,1,1,0,1,1],[1,0,0,0,1,1,0,1,1]]) # Missing class at index 2 and extra class at index 7
    p = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.92,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)
    losses, classes = groupLasso(noisy_y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def sortT(n,alpha,beta):
    noisy_y = tf.constant([[1,0,0,0,1,1,0,1,1],[1,0,0,0,1,1,0,1,1],[1,0,1,1,1,1,0,1,1]]) 
    p = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.92,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)
    losses, classes= groupLasso(noisy_y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def pure_missT(n,alpha,beta):
    noisy_y = tf.constant([[1,0,0,0,1,0,0,0,1]]) # Missing classes at indexes 2 and 5
    p = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)
    losses, classes = groupLasso(noisy_y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def pure_extraT(n,alpha,beta):
    noisy_y = tf.constant([[1,0,1,1,1,1,0,1,1]]) # Extra classes at indexes 3 and 7
    p = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)
    losses, classes = groupLasso(noisy_y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def mix_miss_extraT(n,alpha,beta):
    noisy_y = tf.constant([[1,0,0,0,1,1,0,1,1]]) # Missing class at index 2 and extra class at index 7
    p = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)
    losses, classes = groupLasso(noisy_y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def no_noise_labelT(n,alpha,beta):
    y = tf.constant([[1,0,1,0,1,1,0,0,1]]) # 9 CLASSES
    p = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)
    losses, classes = groupLasso(y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def wrongResultT(n,alpha,beta):
    y = tf.constant([[1,0,1,0,1,1,0,0,1]]) # 9 CLASSES
    p = tf.constant([[0.53,0.47,0.58,0.39,0.51,0.54,0.49,0.46,0.61]], dtype=np.float32)
    losses, classes = groupLasso(y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def margT(n,alpha,beta):
    y = tf.constant([[1,0,1,0,1,1,0,0,1]]) # 9 CLASSES
    p = tf.constant([[0.6,0.3,0.6,0.3,0.6,0.6,0.3,0.3,0.6]], dtype=np.float32)
    losses, classes = groupLasso(y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def marginalT(n,alpha,beta):
    '''
    The error function expects a difference of at least 0.5 between 1 and 0 labels.
    Look at 0.27 value in p.
    '''
    y = tf.constant([[1,0,1,0,1,1,0,0,1]]) # 9 CLASSES
    p = tf.constant([[0.76,0.27,0.76,0.25,0.76,0.76,0.25,0.25,0.76]], dtype=np.float32)
    losses, classes = groupLasso(y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def class_choosing(n,alpha,beta):
    y = tf.constant([[1,0,1,1,1,1,0,1,1]])
    p = tf.constant([[.9,.1,.8,.1,.9,.7,.1,.2,.9]], dtype=np.float32)
    losses, classes = groupLasso(y,p,alpha,beta)
    print(f"losses: {losses}")
    print(f"classes: {classes}")

def agreedFlipT(n,alpha,beta):
    noisy_y1 = tf.constant([[1,0,0,0,1,1,0,1,1],[1,0,0,0,1,1,0,1,1],[1,0,1,1,1,1,0,1,1]]) 
    p1 = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.92,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)
    losses1, classes1 = groupLasso(noisy_y1,p1,alpha,beta)
    print(f"losses1: {losses1}")
    print(f"classes1: {classes1}")
    
    noisy_y2 = tf.constant([[1,0,0,0,1,1,0,1,1],[1,0,1,0,1,0,0,0,0],[1,0,1,1,1,1,0,1,1]]) 
    p2 = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.92,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)
    losses2, classes2 = groupLasso(noisy_y2,p2,alpha,beta)
    print(f"losses2: {losses2}")
    print(f"classes2: {classes2}")

    noisy_sample, noisy_class = choose_noisy_classes(losses1, losses2, classes1, classes2, n)
    print(f"noisy_sample: {noisy_sample}")
    print(f"noisy_class: {noisy_class}")

    flipped_y = flip_classes(noisy_y1, noisy_sample, noisy_class)
    print(f"flipped_y: {flipped_y}")

def tests():
    num_flips = 1
    alpha = 1.0
    beta = 1.0
    agreedFlipT(num_flips,alpha,beta)

# tests()
