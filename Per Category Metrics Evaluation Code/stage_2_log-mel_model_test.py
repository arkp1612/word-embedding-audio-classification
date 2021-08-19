#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import urllib,requests
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils

os.environ["CUDA_VISIBLE_DEVICES"]="1"


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

def dict_generator():
    req = urllib.request.Request('https://raw.githubusercontent.com/arkp1612/word-embedding-audio-classification/main/sensible_tags_encoding.txt')
    with urllib.request.urlopen(req) as response:
         the_page = response.read()

    all_sensible_tags = the_page.split(b'\n')

    all_sensible_tags = [x.decode('utf-8').split('\t') for x in all_sensible_tags]

    k = [all_sensible_tags[i][0].encode('utf-8') for i in range(len(all_sensible_tags))]
    v = [all_sensible_tags[i][-1].encode('utf-8') for i in range(len(all_sensible_tags))]

    return([k,v])


def collapser_func(features_dict,table):    

    input_tensor = features_dict['tags']
    out = table.lookup(input_tensor)
    features_dict['tags'] = out

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




def _generate_datasets(tfrecords, audio_format, split=None, which_split=None, sample_rate=16000, num_mels=128, batch_size=8, block_length=1, cycle_length=1, shuffle=True, shuffle_buffer_size=10000, window_length=15, window_random=False,top=50,as_tuple=True,repeat=1):

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

        k,v = dict_generator()

        table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(k),
            values=tf.constant(v),),
            default_value=tf.constant('NA'),
            name="class_weight")


        dataset = dataset.map(lambda x:collapser_func(x,table))

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




def generate_datasets_from_dir(tfrecords_dir, audio_format, split=None, which_split=None, sample_rate=16000, num_mels=128, batch_size=8, block_length=1, cycle_length=1, shuffle=True, shuffle_buffer_size=10000, window_length=15, window_random=False, with_tids=None, with_tags=None, merge_tags=None, num_tags=155, num_tags_db=1, default_tags_db=None, default_tags_db_valid=None, repeat=1, as_tuple=True):
    tfrecords = []

    for file in os.listdir(os.path.expanduser(tfrecords_dir)):
        if file.endswith(".tfrecord") and file.split('_')[0] == audio_format:
            tfrecords.append(os.path.abspath(os.path.join(tfrecords_dir, file)))

    train_size = 0.8*len(tfrecords)
    valid_size = 0.1*len(tfrecords)
    test_size = 0.1*len(tfrecords)

    dataset = []    
    dataset_final_test = []

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



    return dataset_final_test

class AUCPerLabel(tf.keras.metrics.AUC):

    def __init__(self,
                 num_thresholds=200,
                 curve='ROC',
                 summation_method='interpolation',
                 name=None,
                 dtype=None,
                 thresholds=None,
                 label_weights=None):
        super(AUCPerLabel, self).__init__(
            num_thresholds=num_thresholds,
            curve=curve,
            summation_method=summation_method,
            name=name,
            dtype=dtype,
            thresholds=thresholds,
            multi_label=True,
            label_weights=label_weights
        )

    def result_per_label(self):
        per_label = True
        if (self.curve == metrics_utils.AUCCurve.PR and
                self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            return self.interpolate_pr_auc_per_label()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = math_ops.div_no_nan(self.true_positives,
                                     self.true_positives + self.false_negatives)
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = math_ops.div_no_nan(self.false_positives,
                                          self.false_positives + self.true_negatives)
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = math_ops.div_no_nan(
                self.true_positives, self.true_positives + self.false_positives)
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = math_ops.minimum(y[:self.num_thresholds - 1], y[1:])
        else:  # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
            heights = math_ops.maximum(y[:self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        riemann_terms = math_ops.multiply(x[:self.num_thresholds - 1] - x[1:],
                                          heights)
        by_label_auc = math_ops.reduce_sum(
            riemann_terms, name=self.name + '_by_label', axis=0)

        if self.label_weights is None:
            if per_label:
                return by_label_auc
            else:
                # Unweighted average of the label AUCs.
                return math_ops.reduce_mean(by_label_auc)
        else:
            return math_ops.div_no_nan(
                math_ops.multiply(by_label_auc, self.label_weights),
                math_ops.reduce_sum(self.label_weights),
                name=self.name)

    def interpolate_pr_auc_per_label(self):
        num_labels = self.true_positives.shape[1]
        pr_auc_scores = []
        for i in range(num_labels):
            true_positives = self.true_positives[:, i]
            false_positives = self.false_positives[:, i]
            false_negatives = self.false_negatives[:, i]

            dtp = true_positives[:self.num_thresholds -
                                       1] - true_positives[1:]
            p = true_positives + false_positives
            dp = p[:self.num_thresholds - 1] - p[1:]
            prec_slope = math_ops.div_no_nan(
                dtp, math_ops.maximum(dp, 0), name='prec_slope')
            intercept = true_positives[1:] - math_ops.multiply(prec_slope, p[1:])

            safe_p_ratio = array_ops.where(
                math_ops.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0),
                math_ops.div_no_nan(
                    p[:self.num_thresholds - 1],
                    math_ops.maximum(p[1:], 0),
                    name='recall_relative_ratio'),
                array_ops.ones_like(p[1:]))

            pr_auc_increment = math_ops.div_no_nan(
                prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)),
                math_ops.maximum(true_positives[1:] + false_negatives[1:], 0),
                name='pr_auc_increment')

            by_label_auc = math_ops.reduce_sum(
                pr_auc_increment, name=self.name + '_by_label', axis=0)
            if self.label_weights is None:
                # Evenly weighted average of the label AUCs.
                pr_auc_scores.append(math_ops.reduce_mean(by_label_auc, name=self.name).numpy())
            else:
                # Weighted average of the label AUCs.
                pr_auc_scores.append(math_ops.div_no_nan(
                    math_ops.reduce_sum(
                        math_ops.multiply(by_label_auc, self.label_weights)),
                    math_ops.reduce_sum(self.label_weights),
                    name=self.name).numpy())
        return tf.constant(pr_auc_scores)

    def get_config(self):
        config = super(AUCPerLabel, self).get_config()
        return config


if __name__== "__main__":
    print("Creating datasets")
    test_ds = generate_datasets_from_dir('/srv/data/tfrecords/log-mel-complete','log-mel-spectrogram')
    print("Datasets built")
    log_dir = os.getcwd()
    log_dir = os.path.join(os.path.expanduser(log_dir), 'log-mel-spectogram_stage_2',)

    filepath = os.path.join(log_dir, 'mymodel.h5')

    model = tf.keras.models.load_model(filepath)
 
    metric_1 = AUCPerLabel(name='ROC_AUC',curve='ROC',dtype=tf.float32)
    metric_2 = AUCPerLabel(name='PR_AUC',curve='PR',dtype=tf.float32)

    for entry in tqdm(test_ds):
        audio_batch, label_batch = entry[0], entry[1]
        logits = model(audio_batch, training=False)
        metric_1.update_state(label_batch,logits)
        metric_2.update_state(label_batch, logits)
    
    all_tags = ['rock', 'pop', 'indie', 'electronic', 'dance', 'alternative rock', 
                'jazz', 'singer-songwriter', 'metal', 'chillout', 'classic rock', 
    		'soul', 'indie rock', 'electronica', 'folk', 'instrumental', 
   		'punk', 'oldies', 'mellow', 'sexy', 'loved', 'sad', 'happy', 
   		'good', 'romantic', 'melancholic', 'great', 'dark', 'dreamy','hot', 
   		'energetic', 'calm', 'funny', 'haunting', 'intense', 'alternative', 
   		'beautiful', 'awesome', 'british', 'chill', 'american', 'cool', 
   		'favorite', 'acoustic', 'party', '2000s', '80s', '90s', '60s', '70s']
    
    directory = os.path.join(os.getcwd(),'test')
    
    with open(os.path.join(directory,'stage_2_model_log-mel_per_cat_test.txt'), 'w') as f:
        f.write('tags: {} ROC_AUC: {} ; PR_AUC: {}'.format(all_tags,np.round(metric_1.result_per_label().numpy()*100, 2), np.round(metric_2.result_per_label().numpy()*100, 2)))
 
