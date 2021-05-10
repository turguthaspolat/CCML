'''
File: main.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de) 
'''

# Standard library imports
import os

# Third party imports
import tensorflow as tf
import numpy as np
import argparse
from tensorflow import keras

# Local imports
from run_together import run_together
from model.model import Model
from model.modelBuilder import build_model
from data_prep.prep_tf_records import load_archive, load_ucmerced_set
from utils.arguments import CoTrArgs

def add_arguments():
    ap = argparse.ArgumentParser(prog='Consensual Collaborative Multi-label Learning', 
            description='This is the code for the Bachelor"s Thesis (Collaborative Learning Models \
                for Classification of Remote Sensing Images with Noisy Labels) of Ahmet Kerem Aksoy \
            	at Technical University of Berlin. Date: November 10, 2020.')
    ap.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size. Default is 32.')
    ap.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs. Default is 10.')
    ap.add_argument('-a', '--architecture', default='resnet', help='Possible architectures: \
        resnet, denseNet, SCNN, paper_model, keras_model, modified_SCNN, batched_SCNN. Default is resnet.')
    ap.add_argument('-d', '--dataset_path', help='Give the path to the folder where tf.record files are located.')
    ap.add_argument('-ch', '--channels', default='RGB', help='Decide what channels you want to load: \
        RGB or ALL. Default is RGB.')
    ap.add_argument('-lb', '--label_type', default='BEN-19', help='Decide what version of BigEarthNet \
        you want to load, or load UC Merced Land Use data set. Possible values are \
            BEN-12, BEN-19, BEN-43, ucmerced.')
    ap.add_argument('-si', '--sigma', type=float, help='The value of the sigma for the gaussian kernel.')
    ap.add_argument('-sw', '--swap', type=int, help='Swap information between models if 1; if 0 do not swap')
    ap.add_argument('-sr', '--swap_rate', type=float, help='The percentage of the swap between the two models.')
    ap.add_argument('-lto', '--lambda_two', type=float, help='Strength of discrepancy component. \
        Give a value between 0. and 1.')
    ap.add_argument('-ltr', '--lambda_three', type=float, help='Strength of consistency component. \
        Give a value between 0. and 1.')
    ap.add_argument('-fb', '--flip_bound', type=float, help='The percentage of unflipped training epochs')
    ap.add_argument('-fp', '--flip_per', type=float, help='Class flipping percentage')
    ap.add_argument('-ma', '--miss_alpha', type=float, help='Rate of miss noise to be included into the error loss')
    ap.add_argument('-eb', '--extra_beta', type=float, help='Rate of extra noise to be included into the error loss')
    ap.add_argument('-an', '--add_noise', type=int, help='Add random label noise to dataset. Enter 0 for no noise; \
        enter 1 for adding noise.')
    ap.add_argument('-nty', '--noise_type', type=int, default=1, help='Choose the noise type to be added to \
        the dataset. Possible values are 1 for Random Noise per Sample and 2 for Mix Label Noise.')
    ap.add_argument('-sar', '--sample_rate', type=float, help='Percentage of samples in a mini-batch \
        to be noisified. Give a value between 0. and 1.')
    ap.add_argument('-car', '--class_rate', type=float, help='Percentage of labels in a sample to be noisified. \
        Mix Label Noise does not use this. Give a value between 0. and 1.')
    ap.add_argument('-dm', '--divergence_metric', help='Possible metrics to diverge and converge the models: \
        mmd, shannon, wasserstein, nothing')
    ap.add_argument('-alp', '--alpha', type=float, help='How much of error_loss_array do you want? \
        Give a value between 0. and 1.')
    ap.add_argument('-test', '--test', type=int, default=0, help='Enter 1 to test the model using \
        only a small portion of the datasets. Default is 0.')
    args = vars(ap.parse_args())

    return args

def main(args):

    tf.random.set_seed(0)
    np.random.seed(0)

    # Choose which version to use
    if args['label_type'] == 'BEN-19':
        NUM_OF_CLASSES = 19
    elif args['label_type'] == 'BEN-12':
        NUM_OF_CLASSES = 12
    elif args['label_type'] == 'BEN-43':
        NUM_OF_CLASSES = 43
    elif args['label_type'] == 'ucmerced':
        NUM_OF_CLASSES = 17
    
    # Load the dataset
    if args['label_type'] == 'ucmerced':
        train_dataset, val_dataset, test_dataset = load_ucmerced_set(args["dataset_path"], args['batch_size'], 
                                                                     args['test'])
    elif args['dataset_path']:  
        train_dataset = load_archive(args['dataset_path'] + '/train.tfrecord', NUM_OF_CLASSES, 
                                     args['batch_size'], 1000, args['test'])
        test_dataset = load_archive(args['dataset_path'] + '/test.tfrecord', NUM_OF_CLASSES, 
                                    args['batch_size'], 1000, args['test'])
        val_dataset = load_archive(args['dataset_path'] + '/val.tfrecord', NUM_OF_CLASSES, 
                                   args['batch_size'], 1000, args['test'])
    else:
        raise ValueError('Argument Error: Give the path to the folder where tf.record files are located.')
    
    # Set model1 parameters
    model1 = Model('model1', NUM_OF_CLASSES, args['batch_size'], args['epochs'])
    # Set model2 parameters
    model2 = Model('model2', NUM_OF_CLASSES, args['batch_size'], args['epochs'])
    models = [model1, model2]
  
    # Build the model according to the choosen bands and architecture
    build_model(args['channels'], args['architecture'], model1, model2, args['label_type'])
    
    # Abstractize the arguments
    co_tr_args = CoTrArgs(model1.model, model2.model, train_dataset, test_dataset, 
            val_dataset, args['epochs'], args['batch_size'], args['sigma'], 
            args['swap'], args['swap_rate'], args['lambda_two'], args['lambda_three'], 
            args['flip_bound'], args['flip_per'], args['miss_alpha'], 
            args['extra_beta'], args['add_noise'], args['sample_rate'], 
            args['class_rate'], args['divergence_metric'], args['alpha'], args['label_type'], 
            NUM_OF_CLASSES, args['channels'], args['noise_type'])

    # Run the training loop on both models
    run_together(co_tr_args, model1, model2)

    # Summarize models
    for model_  in models:    
        model_sum = model_.model.summary()
        print(f'model summary: {model_sum}')
        model_.saveModel('last', args['sample_rate'])  

if __name__ == '__main__':
    args = add_arguments()
    main(args)
