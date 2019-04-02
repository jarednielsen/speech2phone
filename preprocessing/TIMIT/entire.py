r"""Preprocessing module for TIMIT data. Defines functions for loading entire audio samples from TIMIT.

Run this command to convert the LDC sphere files to .wav:

    find . -name '*.WAV' -exec sph2pipe -f wav {} {}.wav \;

sph2pipe is available online from the LDC.

Kyle Roth. 2019-02-05.
"""


from os import path, makedirs
from glob import iglob as glob
import warnings
import pickle
import inspect

import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

from speech2phone.preprocessing.TIMIT.phones import _get_dataset_path, get_indices, to_onehot


def _load_from_dir(directory, max_files=None):
    """Load the dataset from the specified directory.

    Warn if a WAV file is encountered without a corresponding PHN file. See module docstring for instruction to
    convert from 'NIST' format to .wav.

    Args:
        directory (str): directory of dataset to load.
        max_files (int): the maximum number of files to load from. Used to create the 'toy' dataset.

    Returns:
        list(np.ndarray): NumPy arrays of audio data.
        list(np.ndarray): Array of indices in the audio corresponding to phoneme boundaries.
        list(np.ndarray): Array of phoneme indices corresponding to the audio data.
    """
    samples = []
    bounds = []
    phonemes = []

    file_list = glob(path.join(directory, '**/*.WAV.wav'), recursive=True)
    if max_files is not None:
        file_list = list(file_list)[:max_files]

    for file in file_list:
        if path.isfile(file[:-7] + 'PHN'):
            # read entire audio file
            try:
                _, entire = wavfile.read(file)  # no need to store the sample rate
                samples.append(entire)
            except ValueError as e:
                raise ValueError('file audio could not be read: {}\n{}'.format(file, str(e)))

            # get each phoneme from audio, according to .PHN file
            with open(file[:-7] + 'PHN') as phn:
                temp_bounds = []
                temp_phones = []
                for line in phn:
                    left, right, phone = line.split()
                    temp_bounds.append([int(left), int(right)])
                    temp_phones.append(phone)
                bounds.append(np.array(temp_bounds))
                phonemes.append(get_indices(temp_phones))  # convert to indices
        else:
            warnings.warn('wav file has no phn file: {}'.format(file))
    return samples, bounds, phonemes


def get_data(dataset='train', preprocessor=None, batch_preprocess=True, TIMIT_root='TIMIT/TIMIT/',
             use_cache=True, y_type='categorical'):
    """Return the train, val, or test set from the TIMIT directory.

    If batch_preprocess is set, the preprocessor must accept a list of data points (audio samples) and a list of
    corresponding labels (phoneme strings). Otherwise, it must accept a single data point and its corresponding
    label (phoneme string). In either case, it should return preprocessed versions of both inputs.

    The train and val sets are differentiated by using the same random seed for splitting with sklearn's
    train_test_split function.

    Args:
        dataset (str): specifies the requested dataset; one of {'train', 'val', 'test', 'toy'}.
        preprocessor (callable): preprocessing function to be applied to data. Call signature must allow (x, b, y)
                                 where x is a single np.ndarray of audio, b is an np.ndarray of boundaries
                                 (shape (2,)), and y is a label (str). If batch_preprocess is True, preprocessor is
                                 called on X, bounds, y where X is a np.ndarray of all the audio, bounds is an
                                 np.ndarray of boundaries (shape (n, 2)), and y is a list of labels.
        batch_preprocess (bool): if True, preprocessor is called on the entire dataset at once. Otherwise, preprocessor
                                 is called on a single data point and label at a time.
        TIMIT_root (str): specifies the root data directory of the TIMIT corpus. Should contain subdirectories 'TRAIN'
                          and 'TEST'.
        use_cache (bool): if True, reuses preprocessed data cached in TIMIT_root/cache if available. If False, recreates
                          dataset and caches it in that location.
        y_type (str): the type of label set to return; one of {'categorical', 'one-hot'}.

    Returns:
        list(np.ndarray): audio data, preprocessed as specified.
        list(np.ndarray): Array of indices in the audio corresponding to phoneme boundaries.
        list(np.ndarray): arrays of phonemes corresponding to each audio file.
    """
    if y_type.lower() not in ('categorical', 'one-hot'):
        raise ValueError('y_type must be one of (\'categorical\', \'one-hot\')')

    # specify the directory according to the dataset being used
    set_root = _get_dataset_path(TIMIT_root, dataset)

    # get the name of the preprocessing function to see if it's been used before
    if preprocessor is None:
        fn_name = 'none'
    else:
        fn_name = dict(inspect.getmembers(preprocessor))['__name__']

    # ensure the caching directory is available
    pickle_path = path.join(TIMIT_root, 'cache/entire-{}/{}.pkl'.format(dataset.lower(), fn_name))
    makedirs(path.join(TIMIT_root, 'cache/entire-{}'.format(dataset.lower())), exist_ok=True)

    # load data from either cache or directory
    if use_cache and path.isfile(pickle_path):  # cache exists
        print('Loading {}/{} set from cache...'.format(dataset.lower(), fn_name), end='', flush=True)
        with open(pickle_path, 'rb') as infile:
            X, bounds, y = pickle.load(infile)
        print(' done.')
    else:  # not cached
        print('Loading {} set from files...'.format(dataset.lower()), end='', flush=True)
        # load from files
        if dataset.lower() == 'toy':
            X, bounds, y = _load_from_dir(set_root, max_files=100)
        else:
            X, bounds, y = _load_from_dir(set_root)
        print(' done.')

        # get just train set or just val set if necessary
        if dataset.lower() == 'train':
            X, _, bounds, _, y, _ = train_test_split(X, bounds, y, test_size=0.25, random_state=42)
        elif dataset.lower().startswith('val'):
            _, X, _, bounds, _, y = train_test_split(X, bounds, y, test_size=0.25, random_state=42)

        # apply preprocessor
        if preprocessor:
            print('Applying preprocessor "{}"...'.format(fn_name), end='', flush=True)
            if batch_preprocess:
                X, bounds, y = preprocessor(X, bounds, y)
            else:
                X, y = zip(*(preprocessor(x, b, wai) for x, b, wai in zip(X, bounds, y)))
                X, y = list(X), list(y)
            print(' done.')

        # cache the dataset for future use
        print('Saving {}/{} set to cache...'.format(dataset.lower(), fn_name), end='', flush=True)
        with open(pickle_path, 'wb+') as outfile:
            pickle.dump((X, bounds, y), outfile)
        print(' done.')

    # convert to one-hot if necessary
    if y_type.lower() == 'one-hot':
        y = to_onehot(y)

    return X, bounds, y


def test_TIMIT_entire():
    """Test get_data using default parameters."""
    X, bounds, y = get_data()
    print("running test_TIMIT_entire()")
    print('Object lengths are:', len(X), len(bounds), len(y))
    print('Shapes of first elements are:', X[0].shape, bounds[0].shape, y[0].shape)
