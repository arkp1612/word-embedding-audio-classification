#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import urllib,requests
import numpy as np


def stage_2_model():
    
    #------------------Genreating datasets-----------------------
    def _waveform_parse_function(example_proto,feature_description):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        parsed_features['audio'] = tf.reshape(tf.sparse.to_dense(parsed_features['audio']),[-1,])
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
        v = [all_sensible_tags[i][0].encode('utf-8') for i in range(len(all_sensible_tags))]

        return([k,v])
    
    def collapser_func(features_dict):
        
        k,v = dict_generator()
                
        table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(k),
            values=tf.constant(v),
        ),
        default_value=tf.constant('NA'),
        name="class_weight")

        input_tensor = features_dict['tags']
        out = table.lookup(input_tensor)
        features_dict['tags'] = out

        return features_dict
    
    def hot_encoder(features_dict,top):
        tag_to_tag_num = generate_tag_num_dict()

        k = [x[1] for x in tag_to_tag_num]
        v = [x[0] for x in tag_to_tag_num]


        table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(k),
            values=tf.constant(v),
        ),
        default_value=tf.constant(999),
        name="class_weight")

        input_tensor = features_dict['tags']
        idxs = tf.cast(table.lookup(input_tensor),tf.int64)
        idxs = tf.boolean_mask(idxs,tf.math.less(idxs, top))
        hot_encodes = tf.one_hot(idxs,depth=top, on_value=1, off_value=0)
        hot_encodes_as_vector = tf.reshape(hot_encodes, [-1])
        zero_padding = tf.zeros([top * top] - tf.shape(hot_encodes_as_vector), dtype=hot_encodes.dtype)
        hot_encodes_padded = tf.concat([hot_encodes_as_vector, zero_padding], 0)
        features_dict['tags'] = tf.reshape(hot_encodes_padded, [top, top])

        return features_dict
    def generate_tag_num_dict():
    
        all_tags = ['rock', 'female', 'pop', 'alternative', 'male', 'indie', 
                'electronic', '00s', 'rnb', 'dance', 'hip-hop', 'instrumental', 
                'chillout', 'alternative rock', 'jazz', 'metal', 'classic rock', 
                'indie rock', 'rap', 'soul', 'mellow', '90s', 'electronica', '80s', 
                'folk', 'chill', 'funk', 'blues', 'punk', 'hard rock', 'pop rock', 
                '70s', 'ambient', 'experimental', '60s', 'easy listening', 
                'rock n roll', 'country', 'electro', 'punk rock', 'indie pop', 
                'heavy metal', 'classic', 'progressive rock', 'house', 'ballad', 
                'psychedelic', 'synthpop', 'trance', 'trip-hop', 'lounge', 
                'techno', 'post-punk', 'reggae', 'new wave', 'britpop', 
                'blues rock', 'folk rock', 'death metal', 'emo', 'soft rock', 
                'latin', 'electropop', 'progressive', '50s', 'disco', 'industrial', 
                'progressive metal', 'post-rock', 'smooth jazz', 'pop punk', 
                'metalcore', 'thrash metal', 'gothic', 'psychedelic rock', 
                'alt-country', 'club', 'alternative  punk', 'avant-garde', 'ska', 
                'americana', 'nu jazz', 'fusion', 'post-hardcore', 'new age', 
                'power pop', 'nu metal', 'black metal', 'power metal', 'grunge', 
                'acid jazz', 'dub', 'garage rock', 'neo-soul', 
                'melodic death metal', 'underground hip-hop', 'alternative metal', 
                'idm', 'darkwave', 'alt rock', 'gothic metal', 'ethereal', 'swing', 
                'glam rock', 'progressive trance', 'lo-fi', 'rockabilly', 'classical', 
                'metro downtempo', 'dream pop', 'melodic metal', 'doom metal', 'bass', 
                'shoegaze', 'gothic rock', 'heavy', 'dancehall', 'art rock', 
                'classic country', 'screamo', 'christmas', 'hardcore punk', 
                'celtic', 'garage', 'rockpop', 'synth', 'indietronica', 
                'vocal jazz', 'jazz fusion', 'stoner rock', 'jazz vocal', 
                'electro house', 'grindcore', 'vocal trance', 'christian rock', 
                'indie folk', 'ebm', 'old school soul', 'goth', 'southern rock', 
                'progressive house', 'symphonic metal', 'eurodance', 'deep house', 
                'roots reggae', 'gospel', 'industrial metal', 'brutal death metal', 
                'bluegrass', 'minimal techno', 'electroclash', 'salsa', 
                'speed metal', 'thrash', 'experimental rock']

        all_tags = [x.encode('utf-8') for x in all_tags]

        return(list(enumerate(all_tags)))
    
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
    
        tfrecords = np.array(tfrecords, dtype=np.unicode) # allow for single str as input
        tfrecords = np.vectorize(lambda x: os.path.abspath(os.path.expanduser(x)))(tfrecords) # fix issues with relative paths in input list
        
        if split:
            if np.sum(split) == 100:
                np_split = np.cumsum(split) * len(tfrecords) // 100
            else:
                assert np.sum(split) <= len(tfrecords) , 'split exceeds the number of available .tfrecord files'
                np_split = np.cumsum(split)
            tfrecords_split = np.split(tfrecords, np_split)
            tfrecords_split = tfrecords_split[:-1] # discard last empty split
        else:
            tfrecords_split = [tfrecords]

        datasets = []

        for files_list in tfrecords_split:
            if files_list.size > 1: # read files in parallel (number of parallel threads specified by cycle_length)
                files = tf.data.Dataset.from_tensor_slices(files_list)
                dataset = files.interleave(tf.data.TFRecordDataset, block_length=block_length, cycle_length=cycle_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            else:
                dataset = tf.data.TFRecordDataset(files_list)

            # parse serialized features
            if audio_format == 'waveform':
                dataset = dataset.map(lambda x: _waveform_parse_function(x, AUDIO_FEATURES_DESCRIPTION), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            else:
                dataset = dataset.map(lambda x: _spectrogram_parse_function(x, AUDIO_FEATURES_DESCRIPTION), num_parallel_calls=tf.data.experimental.AUTOTUNE)
       
                
            # shuffle
            if shuffle:
                dataset = dataset.shuffle(shuffle_buffer_size)
                
                
            #clean, collapse and hot encode
            dataset = dataset.map(lambda x:cleaner_func(x))
            dataset = dataset.map(lambda x:collapser_func(x))
            dataset = dataset.map(lambda x:hot_encoder(x,top))
            
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

            datasets.append(dataset)

        if split:
            datasets = np.where(np.array(split) != 0, datasets, None) # useful when split contains zeros

        if which_split is not None:
            if split is not None:
                assert len(which_split) == len(split), 'split and which_split must have the same length'
                datasets = np.array(datasets)[np.array(which_split, dtype=np.bool)].tolist()
            else:
                datasets = datasets + [None] * (which_split.count(1) - 1) # useful when trying to unpack datasets (if you need a fixed number of datasets), but split has not been provided

        if len(datasets) == 1:
            return datasets[0]
        else:
            return datasets
            

    
    
    def generate_datasets_from_dir(tfrecords_dir, audio_format, split=None, which_split=None, sample_rate=16000, num_mels=96, batch_size=32, block_length=1, cycle_length=1, shuffle=True, shuffle_buffer_size=10000, window_length=15, window_random=False, with_tids=None, with_tags=None, merge_tags=None, num_tags=155, num_tags_db=1, default_tags_db=None, default_tags_db_valid=None, repeat=1, as_tuple=True):
        ''' Reads the TFRecords from the input directory and produces a list tf.data.Dataset objects ready for training/evaluation.

        Parameters:
        ----------
        tfrecords_dir: str
            Directory containing the .tfrecord files.
        split: tuple
            Specifies the number of train/validation/test files to use when reading the .tfrecord files.
            If values add up to 100, they will be treated as percentages; otherwise, they will be treated as actual number of files to parse.
        which_split: tuple
            Applies boolean mask to the datasets obtained with split. Specifies which datasets are actually returned.
        sample_rate: int
            Specifies the sample rate used to process audio tracks.
        batch_size: int
            Specifies the dataset batch_size.
        block_length: int
            Controls the number of input elements that are processed concurrently.
        cycle_length: int
            Controls the number of input elements that are processed concurrently.
        shuffle: bool
            If True, shuffles the dataset with buffer size = shuffle_buffer_size.
        shuffle_buffer_size: int
            If shuffle is True, sets the shuffle buffer size.
        window_length: int
            Specifies the desired window length (in seconds).
        window_random: bool
            Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).

        num_mels: int
            The number of mels in the mel-spectrogram.

        num_tags: int
            The total number of tags.

        num_tags_db: int
            The total number of tags databases used.

        default_tags_db: int
            The index of the tags database that you want to use, when multiple databases are available.
        default_tags_db_valid: int
            The index of the tags database that you want to use for validation/testing, when multiple databases are available.

        with_tids: list
            If not None, contains the tids to be trained on.
        with_tags: list
            If not None, contains the tags to be trained on.
        merge_tags: list
            If not None, contains the lists of tags to be merged together (only applies if with_tags is specified).
        repeat: int
            If not None, repeats the dataset only for a given number of epochs (default is repeat indefinitely).
        as_tuple: bool
            If True, discards tid's and transforms features into (audio, tags) tuples.
        '''

        tfrecords = []

        for file in os.listdir(os.path.expanduser(tfrecords_dir)):
            if file.endswith(".tfrecord") and file.split('_')[0] == audio_format:
                tfrecords.append(os.path.abspath(os.path.join(tfrecords_dir, file)))

        return generate_datasets(tfrecords, audio_format, split=split, which_split=which_split, 
                                 sample_rate = sample_rate, batch_size = batch_size, 
                                 block_length = block_length, cycle_length = cycle_length, shuffle = shuffle, shuffle_buffer_size = shuffle_buffer_size, 
                                 window_length = window_length, window_random = window_random, 
                                 num_mels = num_mels, num_tags = num_tags,
                                 repeat = repeat, as_tuple = as_tuple)                     



