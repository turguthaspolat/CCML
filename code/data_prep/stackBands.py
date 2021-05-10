'''
File: stackBands.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports
import tensorflow as tf

# Stack all bands of BigEarthNet
def get_stacked_bands(inputs):
    return (
        tf.stack([inputs['B04'], inputs['B03'], inputs['B02'], inputs['B08']], axis=3),
        tf.stack([inputs['B05'], inputs['B06'], inputs['B07'], inputs['B8A'],
            inputs['B11'], inputs['B12']], axis=3),
        tf.stack([inputs['B01'], inputs['B09']], axis=3)
    )

# Get the 10 bands that we are interested in, and resize them 
def prepare_input(inputs):
    bands_10m, bands_20m, _ = get_stacked_bands(inputs)

    return tf.concat([
        bands_10m, tf.image.resize(bands_20m, [120, 120], tf.image.ResizeMethod.BICUBIC)], axis=3)
