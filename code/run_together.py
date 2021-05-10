'''
File: run_together.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Standard library imports
from datetime import datetime
import os

# Third party imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from noisifier import Noisifier

# Local imports
from utils.loggers import TrainMetrics, ValMetrics, TestMetrics
from train import train
from evaluate import evaluate
from validate import validate

def run_together(args, model1, model2):

	# Prediction treshold for the classification function 
    pred_treshold = 0.5
    logdir = '../output/logs/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.model1 == None or args.model2 == None:
        raise Exception('Models are not built properly')

    optimizer_1 = keras.optimizers.Adam(learning_rate=0.001)
    optimizer_2 = keras.optimizers.Adam(learning_rate=0.001)

    trainMetrics = TrainMetrics(logdir, args.num_classes)
    valMetrics = ValMetrics(logdir, args.num_classes)
    testMetrics = TestMetrics(logdir, args.num_classes)

    # Create noisifier to add label noise to the dataset
    noisifier = Noisifier()

    for epoch in range(1,args.epochs+1):

        # Variable to be used to determine when to start flipping labels
        start_flipping = epoch / args.epochs
        # Variable used to save example images
        eximage = 1
        
        # Train the model
        train(args, noisifier, pred_treshold, optimizer_1, optimizer_2, trainMetrics, epoch, eximage, start_flipping)
        
        # Validate the model
        validate(args, valMetrics, pred_treshold, epoch, model1, model2)

    # Evaluate the last model
    evaluate(pred_treshold, args.test_data, args.model1, args.model2, testMetrics, args.label_type, args.channels)

    # Load the best model
    load_dir = f'{os.getcwd()}/../saved_models'
    model_path = os.path.join(load_dir, f'best_{str(args.sample_rate).replace(".","_")}_{str(args.batch_size)}.h5')
    best_model = tf.keras.models.load_model(model_path)
    
    # Evaluate the best model
    print('----------TEST RESULTS USING THE BEST MODEL----------')
    evaluate(pred_treshold, args.test_data, best_model, best_model, testMetrics, args.label_type, args.channels)
