# To parse BigEarthNet tfrecord files, the information from the code below is used:
# https://gitlab.tu-berlin.de/rsim/bigearthnet-models-tf/blob/master/BigEarthNet.py 

'''
File: prep_data.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

import tensorflow as tf
from tensorflow import keras
from noisifier import Noisifier

# This only works for single label for now.
def add_noise(y_train, noise_type, noise_rate, nb_class):
    if noise_type != 'none':

        # Add noise to the training data. The last zero is the random seed
        noisifier = Noisifier()
        noised_y_train = noisifier.noisify(y_train, noise_type, noise_rate, nb_class, 0)
        y_train = noised_y_train
    
    return y_train

def parse_function(example_proto, label_type):
    if label_type == 'BigEarthNet-19':
        nb_class = 43
    elif label_type == 'original':
        nb_class == 19

    parsed_features = tf.io.parse_single_example(
            example_proto, 
            {
                'B01': tf.io.FixedLenFeature([20*20], tf.int64),
                'B02': tf.io.FixedLenFeature([120*120], tf.int64),
                'B03': tf.io.FixedLenFeature([120*120], tf.int64),
                'B04': tf.io.FixedLenFeature([120*120], tf.int64),
                'B05': tf.io.FixedLenFeature([60*60], tf.int64),
                'B06': tf.io.FixedLenFeature([60*60], tf.int64),
                'B07': tf.io.FixedLenFeature([60*60], tf.int64),
                'B08': tf.io.FixedLenFeature([120*120], tf.int64),
                'B8A': tf.io.FixedLenFeature([60*60], tf.int64),
                'B09': tf.io.FixedLenFeature([20*20], tf.int64),
                'B11': tf.io.FixedLenFeature([60*60], tf.int64),
                'B12': tf.io.FixedLenFeature([60*60], tf.int64),
                'patch_name': tf.io.VarLenFeature(dtype=tf.string),
                label_type + '_labels': tf.io.VarLenFeature(dtype=tf.string),
                label_type + '_labels_multi_hot': tf.io.FixedLenFeature([nb_class], tf.int64)
            }
        )

    return {
        'B01': tf.reshape(parsed_features['B01'], [20, 20]),
        'B02': tf.reshape(parsed_features['B02'], [120, 120]),
        'B03': tf.reshape(parsed_features['B03'], [120, 120]),
        'B04': tf.reshape(parsed_features['B04'], [120, 120]),
        'B05': tf.reshape(parsed_features['B05'], [60, 60]),
        'B06': tf.reshape(parsed_features['B06'], [60, 60]),
        'B07': tf.reshape(parsed_features['B07'], [60, 60]),
        'B08': tf.reshape(parsed_features['B08'], [120, 120]),
        'B8A': tf.reshape(parsed_features['B8A'], [60, 60]),
        'B09': tf.reshape(parsed_features['B09'], [20, 20]),
        'B11': tf.reshape(parsed_features['B11'], [60, 60]),
        'B12': tf.reshape(parsed_features['B12'], [60, 60]),
        'patch_name': parsed_features['patch_name'],
        label_type + '_labels': parsed_features[label_type + '_labels'],
        label_type + '_labels_multi_hot': parsed_features[label_type + '_labels_multi_hot']
    }

def add_band_stats(dataset, label_type):
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

    B01  = ((dataset['B01'] - BAND_STATS['mean']['B01']) / BAND_STATS['std']['B01']).astype(np.float32)
    B02  = ((dataset['B02'] - BAND_STATS['mean']['B02']) / BAND_STATS['std']['B02']).astype(np.float32)
    B03  = ((dataset['B03'] - BAND_STATS['mean']['B03']) / BAND_STATS['std']['B03']).astype(np.float32)
    B04  = ((dataset['B04'] - BAND_STATS['mean']['B04']) / BAND_STATS['std']['B04']).astype(np.float32)
    B05  = ((dataset['B05'] - BAND_STATS['mean']['B05']) / BAND_STATS['std']['B05']).astype(np.float32)
    B06  = ((dataset['B06'] - BAND_STATS['mean']['B06']) / BAND_STATS['std']['B06']).astype(np.float32)
    B07  = ((dataset['B07'] - BAND_STATS['mean']['B07']) / BAND_STATS['std']['B07']).astype(np.float32)
    B08  = ((dataset['B08'] - BAND_STATS['mean']['B08']) / BAND_STATS['std']['B08']).astype(np.float32)
    B8A  = ((dataset['B8A'] - BAND_STATS['mean']['B8A']) / BAND_STATS['std']['B8A']).astype(np.float32)
    B09  = ((dataset['B09'] - BAND_STATS['mean']['B09']) / BAND_STATS['std']['B09']).astype(np.float32)
    B11  = ((dataset['B11'] - BAND_STATS['mean']['B11']) / BAND_STATS['std']['B11']).astype(np.float32)
    B12  = ((dataset['B12'] - BAND_STATS['mean']['B12']) / BAND_STATS['std']['B12']).astype(np.float32)
    if label_type == 'original':
        multi_hot_label = dataset['original_labels_multi_hot'].astype(np.float32)
    elif label_type == 'BigEarthNet-19':
        dataset['BigEarthNet-19_labels_multi_hot'].astype(np.float32)

    return {'B01': B01,
            'B02': B02,
            'B03': B03,
            'B04': B04,
            'B05': B05,
            'B06': B06,
            'B07': B07,
            'B08': B08,
            'B8A': B8A,
            'B09': B09,
            'B11': B11,
            'B12': B12,
            'patch_name': dataset['patch_name'],
            label_type + '_labels': dataset[label_type + '_labels'],
            label_type + '_labels_multi_hot': multi_hot_label
            }

def read_tfrecord(dataset, batch_size, label_type, nb_epoch):
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.shuffle(buffer_size=130000)
    # NOT SURE IF I WANNA DO THAT
    # dataset = dataset.repeat(nb_epoch)
    dataset = dataset.map(
            lambda x: parse_function(x, label_type),
            num_parallel_calls=10
        )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)

    # dataset = add_band_stats(dataset, label_type)
    
    return dataset

def prep_data(dataset, batch_size, label_type, noise_type, noise_rate, nb_epoch):

    if label_type == 'BigEarthNet-19':
        nb_class = 43
    elif label_type == 'original':
        nb_class = 19

    train_dataset = read_tfrecord(f'{dataset}/train.tfrecord', batch_size, label_type, nb_epoch)
    val_dataset = read_tfrecord(f'{dataset}/val.tfrecord', batch_size, label_type, nb_epoch)
    test_dataset = read_tfrecord(f'{dataset}/test.tfrecord', batch_size, label_type, nb_epoch)

    return train_dataset, test_dataset, val_dataset, nb_class


def test():

    '''
    label type is either gonna be BigEarthNet-19 or original
    '''

    dataset_path = '../data/BEN-tfrecords-small'

    train_dataset, test_dataset, val_dataset, nb_class = prep_data(dataset_path, 128, 'BigEarthNet-19', 'doesna matter', 0.5, 2)

    for step, batch in enumerate(train_dataset):   
        print(batch['BigEarthNet-19_labels_multi_hot'].shape)
        rgb_image = tf.stack([batch['B04'], batch['B03'], batch['B02']], axis=3)
        #print(rgb_image)
        print(f"rgb_image.shape: {rgb_image.shape}")
        print(f"step: {step}")
        break
    
    '''
    for sample_features in train_dataset:
        feature_raw = sample_features['BigEarthNet-19_labels_multi_hot'].numpy()
        print(feature_raw)
        print(feature_raw.shape)
        break
    '''
    '''
    for record in train_dataset.take(1):
        # example = tf.train.Example()
        # example.ParseFromString(record.numpy())
        # print(f"example: {example}")
        print(f"repr(record): {repr(record)}")
    
    print(f"type(train_dataset): {type(train_dataset)}")
    # print(f"type(test_dataset): {type(test_dataset)}")
    # print(f"type(val_dataset): {type(val_dataset)}")
    print(f"nb_class: {nb_class}")
    print(f"train_dataset: {train_dataset}")
    # print(f"test_dataset: {test_dataset}")
    # print(f"val_dataset: {val_dataset}")
    '''
#test()
