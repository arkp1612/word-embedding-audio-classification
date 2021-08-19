#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import urllib,requests
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#------------------Genreating datasets-----------------------
def _waveform_parse_function(example_proto,feature_description):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    parsed_features['audio'] = tf.reshape(tf.sparse.to_dense(parsed_features['audio']),[-1]) #remvove reshape
    parsed_features['tid'] = tf.sparse.to_dense(parsed_features['tid'])
    parsed_features['tags'] = tf.sparse.to_dense(parsed_features['tags'])
    return parsed_features


def _spectrogram_parse_function(example_proto,feature_description):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Log-mel spectrograms have been produced with 128 mel bins
    parsed_features['audio'] = tf.reshape(tf.sparse.to_dense(parsed_features['audio']), [128, -1])
    parsed_features['tid'] = tf.sparse.to_dense(parsed_features['tid'])
    parsed_features['tags'] = tf.sparse.to_dense(parsed_features['tags'])
    return parsed_features

def cleaner_func(features_dict):
    features_dict['tags'] = tf.strings.lower(features_dict['tags'])
    return features_dict


def generate_tag_num_dict():

    all_tags = ['rock', 'pop', 'indie', 'electronic', 'dance', 'alternative rock', 
   'jazz', 'singer-songwriter', 'metal', 'chillout', 'classic rock', 
   'soul', 'indie rock', 'electronica', 'folk', 'instrumental', 
   'punk', 'oldies', 'mellow', 'sexy', 'loved', 'sad', 'happy', 
   'good', 'romantic', 'melancholic', 'great', 'dark', 'dreamy','hot', 
   'energetic', 'calm', 'funny', 'haunting', 'intense', 'alternative', 
   'beautiful', 'awesome', 'british', 'chill', 'american', 'cool', 
   'favorite', 'acoustic', 'party', '2000s', '80s', '90s', '60s', '70s']

    all_tags = [x.encode('utf-8') for x in all_tags]

    return(list(enumerate(all_tags)))    

def hot_encoder(features_dict,table):

    input_tensor = features_dict['tags']
    idxs = tf.cast(table.lookup(input_tensor),tf.int64)
    idxs = tf.boolean_mask(idxs,tf.math.less(idxs, 50))
    features_dict['tags'] = tf.clip_by_value(tf.reduce_max(tf.one_hot(idxs,depth=50, on_value=1, off_value=0),axis=0),0,1)
    return features_dict


def _window_1(features_dict, sample_rate, window_length=15, window_random=False):
    ''' Extracts a window of 'window_length' seconds from the audio tensors (use with tf.data.Dataset.map).
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).
    sample_rate: int
        Specifies the sample rate of the audio track.

    window_length: int
        Length (in seconds) of the desired output window.

    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    slice_length = tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)) # get the actual slice length
    slice_length = tf.reshape(slice_length, ())

    random = tf.constant(window_random, dtype=tf.bool)

    def fn1a(audio, slice_length=slice_length):
        maxval = tf.subtract(tf.shape(audio, out_type=tf.int32)[0], slice_length)
        x = tf.cond(tf.equal(maxval, tf.constant(0)), lambda: tf.constant(0, dtype=tf.int32), lambda: tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32))
        y = tf.add(x, slice_length)
        audio = audio[x:y]
        return audio

    def fn1b(audio):
        mid = tf.math.floordiv(tf.shape(audio, out_type=tf.int32)[0], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensors
        x = tf.subtract(mid, tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32)))
        y = tf.add(mid, tf.math.floordiv(tf.add(slice_length, tf.constant(1)), tf.constant(2, dtype=tf.int32))) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
        audio = audio[x:y]
        return audio

    features_dict['audio'] = tf.cond(random, lambda: fn1a(features_dict['audio']), lambda: fn1b(features_dict['audio']))
    return features_dict

def _window_2(features_dict, sample_rate, window_length=15, window_random=False):
    ''' Extracts a window of 'window_length' seconds from the audio tensors (use with tf.data.Dataset.map).
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).
    sample_rate: int
        Specifies the sample rate of the audio track.

    window_length: int
        Length (in seconds) of the desired output window.

    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    slice_length = tf.math.floordiv(tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)), tf.constant(512, dtype=tf.int32)) # get the actual slice length
    slice_length = tf.reshape(slice_length, ())

    random = tf.constant(window_random, dtype=tf.bool)

    def fn2a(audio, slice_length=slice_length):
        maxval = tf.subtract(tf.shape(audio, out_type=tf.int32)[1], slice_length)
        x = tf.cond(tf.equal(maxval, tf.constant(0)), lambda: tf.constant(0, dtype=tf.int32), lambda: tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32))
        x = tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32)
        y = tf.add(x, slice_length)
        audio = audio[:,x:y]
        return audio

    def fn2b(audio):
        mid = tf.math.floordiv(tf.shape(audio, out_type=tf.int32)[1], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensors
        x = tf.subtract(mid, tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32)))
        y = tf.add(mid, tf.math.floordiv(tf.add(slice_length, tf.constant(1)), tf.constant(2, dtype=tf.int32))) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
        audio = audio[:,x:y]
        return audio

    features_dict['audio'] = tf.cond(random, lambda: fn2a(features_dict['audio']), lambda: fn2b(features_dict['audio']))
    return features_dict

def _window(audio_format):
    ''' Returns the right window function, depending to the specified audio-format. '''

    return {'waveform': _window_1, 'log-mel-spectrogram': _window_2}[audio_format]

def _spect_normalization(features_dict):
    ''' Normalizes the log-mel-spectrograms within a batch. '''

    mean, variance = tf.nn.moments(features_dict['audio'], axes=[1,2], keepdims=True)
    features_dict['audio'] = tf.nn.batch_normalization(features_dict['audio'], mean, variance, offset = 0, scale = 1, variance_epsilon = .000001)
    return features_dict

def _batch_normalization(features_dict):
    ''' Normalizes a batch. '''

    mean, variance = tf.nn.moments(features_dict['audio'], axes=[0])
    features_dict['audio'] = tf.nn.batch_normalization(features_dict['audio'], mean, variance, offset = 0, scale = 1, variance_epsilon = .000001)
    return features_dict

def _tuplify(features_dict, which_tags=None):
    ''' Transforms a batch into (audio, tags) tuples, ready for training or evaluation with Keras. 

    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).
    which_tags: int
        If not None, specifies the database to use (when multiple databases are provided).
    '''

    if which_tags is None:
        return (features_dict['audio'], features_dict['tags'])




def _generate_datasets(tfrecords, audio_format, split=None, which_split=None, sample_rate=16000, num_mels=96, batch_size=32, block_length=1, cycle_length=1, shuffle=True, shuffle_buffer_size=10000, window_length=15, window_random=False,top=50,as_tuple=True,repeat=1):

        AUDIO_FEATURES_DESCRIPTION = {'audio': tf.io.VarLenFeature(tf.float32), 'tags': tf.io.VarLenFeature( tf.string), 'tid': tf.io.VarLenFeature(tf.string)} # tags will be added just below

        assert audio_format in ('waveform', 'log-mel-spectrogram'), 'please provide a valid audio format'
        dataset = tf.data.TFRecordDataset(tfrecords)


        dataset = tf.data.TFRecordDataset(tfrecords)

        # parse serialized features
        if audio_format == 'waveform':
            dataset = dataset.map(lambda x: _waveform_parse_function(x, AUDIO_FEATURES_DESCRIPTION), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(lambda x: _spectrogram_parse_function(x, AUDIO_FEATURES_DESCRIPTION), num_parallel_calls=tf.data.experimental.AUTOTUNE)


        # shuffle
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)


        #clean, collapse and hot encode
        dataset = dataset.map(cleaner_func)


        tag_to_tag_num = generate_tag_num_dict()

        k_num = [x[1] for x in tag_to_tag_num]
        v_num = [x[0] for x in tag_to_tag_num]


        num_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(k_num),
            values=tf.constant(v_num),
        ),
        default_value=tf.constant(999),
        name="class_weight")
        dataset = dataset.map(lambda x:hot_encoder(x,num_table))

        # slice into audio windows
        dataset = dataset.map(lambda x: _window(audio_format)(x, sample_rate, window_length, window_random), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # batch
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # normalize data
        if audio_format == 'log-mel-spectrogram':
            dataset = dataset.map(_spect_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(_batch_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # convert features from dict into tuple
        if as_tuple:
            dataset = dataset.map(lambda x: _tuplify(x, which_tags=None), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        #datasets.append(dataset)


        return dataset


def generate_datasets_from_dir(tfrecords_dir, audio_format, split=None, which_split=None, sample_rate=16000, num_mels=96, batch_size=32, block_length=1, cycle_length=1, shuffle=True, shuffle_buffer_size=10000, window_length=15, window_random=False, with_tids=None, with_tags=None, merge_tags=None, num_tags=155, num_tags_db=1, default_tags_db=None, default_tags_db_valid=None, repeat=1, as_tuple=True):
    tfrecords = []

    for file in os.listdir(os.path.expanduser(tfrecords_dir)):
        if file.endswith(".tfrecord") and file.split('_')[0] == audio_format:
            tfrecords.append(os.path.abspath(os.path.join(tfrecords_dir, file)))

    train_size = 0.8*len(tfrecords)
    valid_size = 0.1*len(tfrecords)
    test_size = 0.1*len(tfrecords)


    dataset = []    
    dataset_final_train = []
    dataset_final_valid = []
    dataset_final_test = []

    for i in range(int(train_size)):
        dataset = _generate_datasets(tfrecords[i], audio_format, split=split, which_split=which_split, 
                             sample_rate = sample_rate, batch_size = batch_size, 
                             block_length = block_length, cycle_length = cycle_length, shuffle = shuffle, shuffle_buffer_size = shuffle_buffer_size, 
                             window_length = window_length, window_random = window_random, 
                             num_mels = num_mels,
                             repeat = repeat, as_tuple = as_tuple)

        if i == 0:
            dataset_final_train = dataset
        else:
            dataset_final_train = dataset_final_train.concatenate(dataset)


    for j in range(int(valid_size)):
        i = j+int(train_size)    
        dataset = _generate_datasets(tfrecords[i], audio_format, split=split, which_split=which_split, 
                             sample_rate = sample_rate, batch_size = batch_size, 
                             block_length = block_length, cycle_length = cycle_length, shuffle = shuffle, shuffle_buffer_size = shuffle_buffer_size, 
                             window_length = window_length, window_random = window_random, 
                             num_mels = num_mels,
                             repeat = repeat, as_tuple = as_tuple)

        if j == 0:
            dataset_final_valid = dataset
        else:
            dataset_final_valid = dataset_final_valid.concatenate(dataset)

    for j in range(int(test_size)):
        i = j+int(train_size)+int(valid_size)  
        dataset = _generate_datasets(tfrecords[i], audio_format, split=split, which_split=which_split, 
                             sample_rate = sample_rate, batch_size = batch_size, 
                             block_length = block_length, cycle_length = cycle_length, shuffle = shuffle, shuffle_buffer_size = shuffle_buffer_size, 
                             window_length = window_length, window_random = window_random, 
                             num_mels = num_mels,
                             repeat = repeat, as_tuple = as_tuple)

        if j == 0:
            dataset_final_test = dataset
        else:
            dataset_final_test = dataset_final_test.concatenate(dataset)



    return [dataset_final_train,dataset_final_valid,dataset_final_test]


def frontend_wave(input):
    ''' Create the frontend model for waveform input. '''

    initializer = tf.keras.initializers.VarianceScaling()
    
    input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 2), name='expdim_1_wave')(input)

    conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=3, padding='valid',
                   activation='relu', kernel_initializer=initializer, name='conv0_wave')(input)
    bn_conv0 = tf.keras.layers.BatchNormalization(name='bn0_wave')(conv0)

    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv1_wave')(bn_conv0)
    bn_conv1 = tf.keras.layers.BatchNormalization(name='bn1_wave')(conv1)
    pool1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool1_wave')(bn_conv1)
    
    conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv2_wave')(pool1)
    bn_conv2 = tf.keras.layers.BatchNormalization(name='bn2_wave')(conv2)
    pool2 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool2_wave')(bn_conv2)

    conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv3_wave')(pool2) 
    bn_conv3 = tf.keras.layers.BatchNormalization(name='bn3_wave')(conv3)
    pool3 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool3_wave')(bn_conv3)

    conv4 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv4_wave')(pool3) 
    bn_conv4 = tf.keras.layers.BatchNormalization(name='bn4_wave')(conv4)
    pool4 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool4_wave')(bn_conv4)
            
    conv5 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv5_wave')(pool4)
    bn_conv5 = tf.keras.layers.BatchNormalization(name='bn5_wave')(conv5)
    pool5 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool5_wave')(bn_conv5)
            
    conv6 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv6_wave')(pool5)
    bn_conv6 = tf.keras.layers.BatchNormalization(name='bn6_wave')(conv6)
    pool6 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool6_wave')(bn_conv6)
    
    exp_dim = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, [3]), name='expdim2_wave')(pool6)
    return exp_dim

def frontend_log_mel_spect(input, y_input=96, num_filts=32):
    ''' Create the frontend model for log-mel-spectrogram input. '''
    
    initializer = tf.keras.initializers.VarianceScaling()
    input = tf.expand_dims(input, 3)
    
    input_pad_7 = tf.keras.layers.ZeroPadding2D(((0, 0), (3, 3)), name='pad7_spec')(input)
    input_pad_3 = tf.keras.layers.ZeroPadding2D(((0, 0), (1, 1)), name='pad3_spec')(input)
    
    # [TIMBRE] filter shape: 0.9y*7
    conv1 = tf.keras.layers.Conv2D(filters=num_filts, 
               kernel_size=[int(0.9 * y_input), 7],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv1_spec')(input_pad_7)    
    bn_conv1 = tf.keras.layers.BatchNormalization(name='bn1_spec')(conv1)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=[int(conv1.shape[1]), 1], 
                      strides=[int(conv1.shape[1]), 1], name='pool1_spec')(bn_conv1)
    p1 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque1_spec')(pool1)
    
    # [TIMBRE] filter shape: 0.9y*3
    conv2 = tf.keras.layers.Conv2D(filters=num_filts*2,
               kernel_size=[int(0.9 * y_input), 3],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv2_spec')(input_pad_3)
    bn_conv2 = tf.keras.layers.BatchNormalization(name='bn2_spec')(conv2)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=[int(conv2.shape[1]), 1], 
                      strides=[int(conv2.shape[1]), 1], name='pool2_spec')(bn_conv2)
    p2 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque2_spec')(pool2)
    
    # [TIMBRE] filter shape: 0.9y*1
    conv3 = tf.keras.layers.Conv2D(filters=num_filts*4,
               kernel_size=[int(0.9 * y_input), 1], 
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv3_spec')(input)
    bn_conv3 = tf.keras.layers.BatchNormalization(name='bn3_spec')(conv3)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=[int(conv3.shape[1]), 1], 
                      strides=[int(conv3.shape[1]), 1], name='pool3_spec')(bn_conv3)
    p3 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque3_spec')(pool3)

    # [TIMBRE] filter shape: 0.4y*7
    conv4 = tf.keras.layers.Conv2D(filters=num_filts,
               kernel_size=[int(0.4 * y_input), 7],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv4_spec')(input_pad_7)
    bn_conv4 = tf.keras.layers.BatchNormalization(name='bn4_spec')(conv4)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=[int(conv4.shape[1]), 1], 
                  strides=[int(conv4.shape[1]), 1], name='pool4_spec')(bn_conv4)
    p4 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque4_spec')(pool4)

    # [TIMBRE] filter shape: 0.4y*3
    conv5 = tf.keras.layers.Conv2D(filters=num_filts*2,
               kernel_size=[int(0.4 * y_input), 3],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv5_spec')(input_pad_3)
    bn_conv5 = tf.keras.layers.BatchNormalization(name='bn5_spec')(conv5)
    pool5 = tf.keras.layers.MaxPool2D(pool_size=[int(conv5.shape[1]), 1], 
                      strides=[int(conv5.shape[1]), 1], name='pool5_spec')(bn_conv5)
    p5 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque5_spec')(pool5)

    # [TIMBRE] filter shape: 0.4y*1
    conv6 = tf.keras.layers.Conv2D(filters=num_filts*4,
               kernel_size=[int(0.4 * y_input), 1],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv6_spec')(input)
    bn_conv6 = tf.keras.layers.BatchNormalization(name='bn6_spec')(conv6)
    pool6 = tf.keras.layers.MaxPool2D(pool_size=[int(conv6.shape[1]), 1], 
                  strides=[int(conv6.shape[1]), 1], name='pool6_spec')(bn_conv6)
    p6 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque6_spec')(pool6)

    # Avarage pooling frequency axis
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=[y_input, 1], 
                             strides=[y_input, 1], name='avgpool_spec')(input)
    avg_pool = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque7_spec')(avg_pool)

    # [TEMPORAL] filter shape: 165*1
    conv7 = tf.keras.layers.Conv1D(filters=num_filts, kernel_size=165,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv7_spec')(avg_pool)
    bn_conv7 = tf.keras.layers.BatchNormalization(name='bn7_spec')(conv7)
    
    # [TEMPORAL] filter shape: 128*1
    conv8 = tf.keras.layers.Conv1D(filters=num_filts*2, kernel_size=128,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv8_spec')(avg_pool)
    bn_conv8 = tf.keras.layers.BatchNormalization(name='bn8_spec')(conv8)

    # [TEMPORAL] filter shape: 64*1
    conv9 = tf.keras.layers.Conv1D(filters=num_filts*4, kernel_size=64,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv9_spec')(avg_pool)
    bn_conv9 = tf.keras.layers.BatchNormalization(name='bn9_spec')(conv9)
    
    # [TEMPORAL] filter shape: 32*1
    conv10 = tf.keras.layers.Conv1D(filters=num_filts*8, kernel_size=32,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv10_spec')(avg_pool)
    bn_conv10 = tf.keras.layers.BatchNormalization(name='bn10_spec')(conv10)
    
    concat = tf.keras.layers.Concatenate(2, name='concat_spec')([p1, p2, p3, p4, p5, p6, bn_conv7, bn_conv8,
                          bn_conv9, bn_conv10])
    
    exp_dim = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 3), name='expdim1_spec')(concat)
    return exp_dim

def backend(input, num_output_neurons, num_units=1024):
    ''' Create the backend model. '''
    
    initializer = tf.keras.initializers.VarianceScaling()
    
    conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[7, int(input.shape[2])],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv1_back')(input)
    bn_conv1 = tf.keras.layers.BatchNormalization(name='bn1_back')(conv1)
    bn_conv1_t = tf.keras.layers.Permute((1, 3, 2), name='perm1_back')(bn_conv1)
    
    bn_conv1_pad = tf.keras.layers.ZeroPadding2D(((3, 3), (0, 0)), name='pad3_1_back')(bn_conv1_t)
    conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[7, int(bn_conv1_pad.shape[2])],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv2_back')(bn_conv1_pad)
    conv2_t = tf.keras.layers.Permute((1,3,2), name='perm2_back')(conv2)
    bn_conv2 = tf.keras.layers.BatchNormalization(name='bn2_back')(conv2_t)
    res_conv2 = tf.keras.layers.Add(name='add1_back')([bn_conv2, bn_conv1_t])
    
    pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], name='pool1_back')(res_conv2)
    
    pool1_pad = tf.keras.layers.ZeroPadding2D(((3, 3), (0, 0)), name='pad3_2_back')(pool1)
    conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[7, int(pool1_pad.shape[2])],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv3_back')(pool1_pad)
    conv3_t = tf.keras.layers.Permute((1, 3, 2), name='perm3_back')(conv3)
    bn_conv3 = tf.keras.layers.BatchNormalization(name='bn3_back')(conv3_t)
    res_conv3 = tf.keras.layers.Add(name='add2_back')([bn_conv3, pool1])
    
    max_pool2 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1), name='glo_max_back')(res_conv3)
    avg_pool2, var_pool2 = tf.keras.layers.Lambda(lambda x: tf.nn.moments(x, axes=[1]), name='moment_back')(res_conv3)
    pool2 = tf.keras.layers.Concatenate(2, name='concat_back')([max_pool2, avg_pool2])
    flat_pool2 = tf.keras.layers.Flatten()(pool2)
    
    flat_pool2_dropout = tf.keras.layers.Dropout(rate=0.5, name='drop1_back')(flat_pool2)
    dense = tf.keras.layers.Dense(units=num_units, activation='relu',
                  kernel_initializer=initializer, name='dense1_back')(flat_pool2_dropout)
    bn_dense = tf.keras.layers.BatchNormalization(name='bn_dense_back')(dense)
    dense_dropout = tf.keras.layers.Dropout(rate=0.5, name='drop2_back')(bn_dense)
    
    return tf.keras.layers.Dense(activation='sigmoid', units=num_output_neurons,
                 kernel_initializer=initializer, name='dense2_back')(dense_dropout)

def build_model(frontend_mode, num_output_neurons=50, y_input=96, num_units=500, num_filts=16, batch_size=32):
    ''' Generate the final model by combining frontend and backend.
    
    Parameters
    ----------
    frontend_mode: {'waveform', 'log-mel-spectrogram'} 
        Specifies the frontend model.
        
    num_output_neurons: int
        The dimension of the prediction array for each audio input. This should
        be set to the length of the a one-hot encoding of tags.
        
    y_input: int, None
        For waveform frontend, y_input will not affect the output of the function.
        For log-mel-spectrogram frontend, this is the height of the spectrogram and should therefore be set as the 
        number of mel bands in the spectrogram.
        
    num_units: int
        The number of neurons in the dense hidden layer of the backend.
        
    num_filts: int
        For waveform, num_filts will not affect the ouput of the function. 
        For log-mel-spectrogram, this is the number of filters of the first CNN layer. See (Pons, et al., 2018) for more details.
    '''

    if frontend_mode not in ('waveform', 'log-mel-spectrogram'):
        raise ValueError("please specify the correct frontend: 'waveform' or 'log-mel-spectrogram'")

    elif frontend_mode == 'waveform':
        input = tf.keras.Input(shape=[None], batch_size=batch_size)
        front_out = frontend_wave(input)

    elif frontend_mode == 'log-mel-spectrogram':
        input = tf.keras.Input(shape=[y_input, None], batch_size=batch_size)
        front_out = frontend_log_mel_spect(input, y_input=y_input, num_filts=num_filts)

    model = tf.keras.Model(input,
                           backend(front_out,
                                   num_output_neurons=num_output_neurons,
                                   num_units=num_units))
    return model

if __name__== "__main__":
    
    train_ds,valid_ds,test_ds = generate_datasets_from_dir('/srv/data/tfrecords/waveform-complete','waveform')
    
    log_dir = os.getcwd()
    log_dir = os.path.join(os.path.expanduser(log_dir), 'waveform_stage_1',)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(log_dir, 'mymodel.h5'),
            monitor = 'val_AUC-PR',
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1,
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir = log_dir,
            histogram_freq = 1,
            write_graph = False,
            update_freq = 1,
            profile_batch = 0, 
        ),

        tf.keras.callbacks.TerminateOnNaN(),

        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_AUC-PR',
            mode = 'max',
            min_delta = 0,
            restore_best_weights = True,
            verbose = 1,
            patience =3
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_AUC-PR',
            mode = 'max',
            factor = 0.5,
            min_delta = 0,
            min_lr = 0,
            verbose = 1,
        ),
    ]

    
    model = build_model('waveform')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM), metrics=[[tf.keras.metrics.AUC(curve='ROC', name='AUC-ROC'), tf.keras.metrics.AUC(curve='PR', name='AUC-PR')]])
    
    history = model.fit(train_ds,validation_data=valid_ds ,verbose = 2,epochs=20,callbacks=callbacks)
    
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(log_dir,'history.json') 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)


    metric_1 = tf.keras.metrics.AUC(name='ROC_AUC',
                                        curve='ROC',
                                        dtype=tf.float32)
    metric_2 = tf.keras.metrics.AUC(name='PR_AUC',
                                        curve='PR',
                                        dtype=tf.float32)

    for entry in tqdm(test_ds):
            audio_batch, label_batch = entry[0], entry[1]
            logits = model(audio_batch, training=False)
            metric_1.update_state(label_batch, logits)
            metric_2.update_state(label_batch, logits)

    directory = os.path.join(os.getcwd(),'test')
    with open(os.path.join(directory,'stage_1_model_waveform_test.txt'), 'w') as f:
        f.write('ROC_AUC: {} ; PR_AUC: {}'.format(np.round(metric_1.result().numpy()*100, 2), np.round(metric_2.result().numpy()*100, 2)))

