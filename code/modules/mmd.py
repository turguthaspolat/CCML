'''
File: mmd.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Standard library imports
from functools import reduce

# Third party imports
import tensorflow as tf

def mmd(X,Y, sigma):

    dimsX = list(X.shape)
    dims_to_reduceX = dimsX[1:] # all dimension except the first (it is the batch_sizes) will be reshaped into a vector
    reduced_dimX = reduce(lambda x,y: x*y, dims_to_reduceX)
   
    dimsY = list(Y.shape)
    dims_to_reduceY = dimsY[1:] # all dimension except the first (it is the batch_sizes) will be reshaped into a vector
    reduced_dimY = reduce(lambda x,y: x*y, dims_to_reduceY)
   
    X = tf.reshape(X, dimsX[:1]+[reduced_dimX])
    Y = tf.reshape(Y, dimsY[:1]+[reduced_dimY])

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.linalg.tensor_diag_part(XX)
    Y_sqnorms = tf.linalg.tensor_diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)
    
    K_XX = tf.exp(-(1./sigma) * (-2. * XX + c(X_sqnorms) + r(X_sqnorms)))
    K_XY = tf.exp(-(1./sigma) * (-2. * XY + c(X_sqnorms) + r(Y_sqnorms)))
    K_YY = tf.exp(-(1./sigma) * (-2. * YY + c(Y_sqnorms) + r(Y_sqnorms)))     
    
    return tf.reduce_mean(K_XX) -2. * tf.reduce_mean(K_XY) + tf.reduce_mean(K_YY)

