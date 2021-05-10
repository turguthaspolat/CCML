#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  File: data.py
  Author: Tristan Kreuziger (tristan.kreuziger@tu-berlin.de)
  Modified by: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
  Created: 2020-07-29 15:18
  Copyright (c) 2020 Tristan Kreuziger under MIT license
"""

# Third party imports
import tensorflow as tf
import pickle

def useTfData(x, y, batch_size, test):

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # Test the model using only a small portion of the datasets
    if test:
        dataset = dataset.take(64)

    SHUFFLE_BUFFER_SIZE = 100
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    dataset = dataset.prefetch(5)
    
    return dataset

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}


def _get_fixed_feature(size):
    """
    Creates a feature with fixed length for the bands.

    Args:
        size (int): the size of the band, e.g. 60 for 60 x 60.

    Returns:
        tf.io.FixedLenFeature: the desired feature.
    """

    return tf.io.FixedLenFeature([size * size], tf.int64)


def _get_feature_description(num_classes):
    """
    Creates a feature description for the images for later parsing.

    Args:
        num_classes (int): the number of classes in the dataset to initialize the multi hot vectors properly.

    Returns:
        dict(str -> tf.io.Feature): the created features mapped by their names.
    """

    return {
        'B01': _get_fixed_feature(20),
        'B02': _get_fixed_feature(120),
        'B03': _get_fixed_feature(120),
        'B04': _get_fixed_feature(120),
        'B05': _get_fixed_feature(60),
        'B06': _get_fixed_feature(60),
        'B07': _get_fixed_feature(60),
        'B08': _get_fixed_feature(120),
        'B8A': _get_fixed_feature(60),
        'B09': _get_fixed_feature(20),
        'B11': _get_fixed_feature(60),
        'B12': _get_fixed_feature(60),
        'BigEarthNet-19_labels_multi_hot': tf.io.FixedLenFeature([num_classes], tf.int64),
        'patch_name': tf.io.VarLenFeature(dtype=tf.string)
    }


def _normalize_band(value, key):
    """
    Normalizes a band with its specific mean and std.

    Args:
        value (tf.Tensor): The band as it has been loaded.
        key (str): The name of the band to look up the statistics.

    Returns:
        tf.Tensor: The normalized values.
    """

    return (tf.cast(value, tf.float32) - BAND_STATS['mean'][key]) / BAND_STATS['std'][key]


def _get_band(feature, name, size):
    """
    Gets a band normalized and correctly scaled from the raw data.

    Args:
        feature (obj): the feature as it was read from the files.
        name (str): the name of the band.
        size (int): the size of the band.

    Returns:
        tf.Tensor: the band parsed into a tensor for further manipulation.
    """

    return _normalize_band(tf.reshape(feature[name], [size, size]), name)


def _transform_example_into_data(parsed_features):
    """
    Transforms the parsed features into tensors.

    Args:
        parsed_features (obj): the examples parsed from file.

    Returns:
        tuple: the x and y portion of the data.
    """

    return (
        {
            'B01': _get_band(parsed_features, 'B01', 20),
            'B02': _get_band(parsed_features, 'B02', 120),
            'B03': _get_band(parsed_features, 'B03', 120),
            'B04': _get_band(parsed_features, 'B04', 120),
            'B05': _get_band(parsed_features, 'B05', 60),
            'B06': _get_band(parsed_features, 'B06', 60),
            'B07': _get_band(parsed_features, 'B07', 60),
            'B08': _get_band(parsed_features, 'B08', 120),
            'B8A': _get_band(parsed_features, 'B8A', 60),
            'B09': _get_band(parsed_features, 'B09', 20),
            'B11': _get_band(parsed_features, 'B11', 60),
            'B12': _get_band(parsed_features, 'B12', 60),
            'patch_name': tf.sparse.to_dense(parsed_features['patch_name'])
        },
        {'labels': tf.cast(parsed_features['BigEarthNet-19_labels_multi_hot'], tf.float32)}
    )


def load_archive(filenames, num_classes, batch_size=0, shuffle_size=0, test=0, num_parallel_calls=10):
    """
    Loads the archive, preprocesses it as needed, and provides it as a batched, prefetched, and shuffled dataset.

    Args:
        filenames (list[str]): the TFRecord filenames to load.
        num_classes (int): the number of classes in this dataset.
        batch_size (int): the size of the batches that will be provided in this dataset. Disabled by setting it to 0.
        shuffle_size (int): the size of the shuffle buffer used when smapling batches. Disabled by setting it to 0.
        num_parallel_calls (int): the number of parallel calls, when loading the data from file.
        prefetch_size (int): the number of elements to prefetch for better throughput.
    """
    def parse_example(example):
        return _transform_example_into_data(tf.io.parse_single_example(example, feature_description))

    # Get the feature description to parse the raw data.
    feature_description = _get_feature_description(num_classes)

    # Load the TFRecord data from file.
    dataset = tf.data.TFRecordDataset(filenames)

    # Test the model using only a small portion of the datasets.
    if test:
        dataset = dataset.take(64)

    # Shuffle the data as the very first step if desired.
    if shuffle_size > 0:
        dataset = dataset.shuffle(shuffle_size)
    
    # Parse all entries.
    dataset = dataset.map(parse_example, num_parallel_calls)
    
    # Exclude the samples that have no labels
    dataset = dataset.filter(lambda x,y : tf.math.not_equal(tf.math.reduce_sum(y['labels']),0))

    # Create a batched dataset.
    if batch_size > 0:
        dataset = dataset.batch(batch_size)

    # Prefetch some of the data to ensure maximum throughput.
    dataset = dataset.prefetch(5)

    return dataset


def load_ucmerced_set(data_path, batch_size, test=0):
    '''
    Helper function to load UC Merced Land Use data set
    '''
    sets_ = []
    for set_ in ['train', 'validation', 'test']:
        x = pickle.load(open(f'{data_path}/x_{set_}.pickle', 'rb'))
        y = pickle.load(open(f'{data_path}/y_{set_}.pickle', 'rb'))
        dataset = useTfData(x, y, batch_size, test)
        sets_.append(dataset)
    
    return sets_[0], sets_[1], sets_[2] 
