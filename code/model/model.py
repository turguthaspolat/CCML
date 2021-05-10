'''
File: model.py 
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Standard library imports
import os

# Third party imports
import numpy as np
import tensorflow as tf    
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, LeakyReLU

class Model:
   
    save_dir = f'{os.getcwd()}/../saved_models'

    def __init__(self, name, classes, batch_size, epochs):
        self.model_name = f'{name}.h5'
        self.classes = classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def __repr__(self):
        return f'Model(self.model_name)'

    def __str__(self):
        return self.model_name

    def build_resnet(self, inputShape):
        model = tf.keras.applications.ResNet50V2(
                    weights=None, include_top=False, input_tensor=tf.keras.Input(shape=inputShape))

        x = Flatten()(model.output)
        output = Dense(self.classes)(x)
        model = keras.Model(inputs=model.input, outputs=[output, model.get_layer('conv2_block3_2_relu').output])

        self.model = model
    
    def build_densenet(self, inputShape):
        model = tf.keras.applications.DenseNet121(
                    weights=None, include_top=False, input_tensor=tf.keras.Input(shape=inputShape))

        x = Flatten()(model.output)
        output = Dense(self.classes)(x)
        model = keras.Model(inputs=model.input, outputs=[output, model.get_layer('conv3_block12_concat').output])

        self.model = model

    def build_SCNN_model(self, inputShape):
        inputs = keras.Input(shape=inputShape)
        x = Conv2D(32, (5,5), padding='same', activation='relu', name='l2-layer')(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        l2_logits = x
        x = Conv2D(32, (5,5), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Flatten()(x)
        outputs = Dense(self.classes)(x)

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='scnn_model')
        self.model = model

    def build_batched_SCNN_model(self, inputShape):
        inputs = keras.Input(shape=inputShape)
        x = Conv2D(32, (5,5), padding='same', activation='relu', name='l2-layer')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        l2_logits = x
        x = Conv2D(32, (5,5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Flatten()(x)
        outputs = Dense(self.classes)(x)

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='scnn_model')
        self.model = model

    def build_modified_SCNN_model(self, inputShape):
        inputs = keras.Input(shape=inputShape)
        x = Conv2D(32, (5,5), padding='same', activation='relu', name='l2-layer')(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(32, (5,5), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        l2_logits = x
        x = Conv2D(32, (5,5), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(32, (5,5), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Flatten()(x)
        outputs = Dense(self.classes)(x)

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='scnn_model')
        self.model = model
    
    def build_paper_model(self, inputShape):
        inputs = keras.Input(shape=inputShape)
        x = Conv2D(128, (3,3), padding='same')(inputs)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3,3), padding='same', name='l2-layer')(x)
        x = LeakyReLU(alpha=0.01)(x)
        l2_logits = x
        x = BatchNormalization()(x) # REMOVE THIS MAYBE???
        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        
        x = Flatten()(x)
        x = Dense(128)(x)
        outputs = Dense(self.classes)(x)

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='dct_model')
    
        self.model = model

    def build_keras_model(self, inputShape):
        inputs = keras.Input(shape=inputShape)
        x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3,3), padding='same', activation='relu', name='l2-layer')(x)
        l2_logits = x
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)
        
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.classes)(x)

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='dct_model')
    
        self.model = model

    def eval(self, X_TEST, Y_TEST):
        if self.model == None:
            raise Exception('The model is not built, call buildModel() method first')
            exit()
        
        val_loss, val_acc = self.model.evaluate(X_TEST, Y_TEST, batch_size=self.batch_size, verbose=1)
        print(f'val_loss is {val_loss} and val_acc is {val_acc}')    

    def saveModel(self, path_appendix, noise_rate):
        if self.model == None:
            raise Exception('The model is not built, call buildModel() method first')
            exit()

        # Save model and weights
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if path_appendix == 'best':
            model_path = os.path.join(self.save_dir, f'{path_appendix}_{str(noise_rate).replace(".","_")}_{str(self.batch_size)}.h5')
        else:
            model_path = os.path.join(self.save_dir, f'{path_appendix}_{str(noise_rate).replace(".","_")}_{self.batch_size}_{self.model_name}')
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)
