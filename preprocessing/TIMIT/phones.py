r"""Preprocessing module for TIMIT data. Defines functions for loading individual phonemes from TIMIT.

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


# mapping from phones to integers, to be used consistently with every dataset
phones = np.array([
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em',
    'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm',
    'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w',
    'y', 'z', 'zh'
])

# mapping from integers to phones, to be used consistently with every dataset
phone_to_idx = {
    phone: i for i, phone in enumerate(phones)
}


def _get_dataset_path(TIMIT_root, dataset):
    """Get the path to the requested dataset.

    Args:
        TIMIT_root (str): path to TIMIT root data directory (e.g. 'TIMIT/TIMIT')
        dataset (str): one of ('train', 'val', 'test', 'toy')
    Returns:
        str: combined path to requested dataset
    """
    if dataset.lower() in {'train', 'val', 'toy'}:
        return path.join(TIMIT_root, 'TRAIN')
    if dataset.lower() == 'test':
        warnings.warn('loading test data; only use test data to demonstrate final results')
        return path.join(TIMIT_root, 'TEST')

    raise ValueError('dataset must be specified as one of (\'train\', \'val\', \'test\', \'toy\')')


def get_phones(indices):
    """Take a vector of indices, and return their respective phonemes.

    Args:
        indices (iterable(int)): vector of indices
    Returns:
        np.ndarray(str): vector of phones
    """
    return phones[indices]


def get_indices(phone_strings):
    """Take a vector of phones, and return their respective indices.

    Args:
        phones (iterable(str)): vector of phones
    Returns:
        np.ndarray(int): vector of indices
    """
    out = []
    for phone in phone_strings:
        out.append(phone_to_idx[phone])
    return np.array(out)


def to_onehot(y):
    """Convert categorical data to one-hot encoding.

    Args:
        y (iterable(int)): vector of integer categorical data
    Returns:
        np.ndarray(int): 2-dimensional encoded version of y
    """
    out = np.zeros((len(y), 61))
    if isinstance(y[0], str):
        # need to convert to indices first
        y = get_indices(np.copy(y))
    # encode
    out[np.arange(len(y)), y] = 1
    return out


def _load_from_dir(directory, padding=0, max_files=None):
    """Load the dataset from the specified directory.

    Warn if a WAV file is encountered without a corresponding PHN file. See module docstring for instruction to
    convert from 'NIST' format to .wav.

    Args:
        directory (str): directory of dataset to load.
        padding (int): the number of audio samples to provide on either side of the phoneme, where available. Default is
                       no padding (0).
        max_files (int): the maximum number of files to load from. Used to create the 'toy' dataset.

    Returns:
        list(np.ndarray): NumPy arrays of audio data.
        list(int): Phoneme indices corresponding to the audio data.
    """
    samples = []
    phonemes = []

    file_list = glob(path.join(directory, '**/*.WAV.wav'), recursive=True)
    if max_files is not None:
        file_list = list(file_list)[:max_files]

    for file in file_list:
        if path.isfile(file[:-7] + 'PHN'):
            # read entire audio file
            try:
                _, entire = wavfile.read(file)  # no need to store the sample rate
            except ValueError as e:
                raise ValueError('file audio could not be read: {}\n{}'.format(file, str(e)))

            # get each phoneme from audio, according to .PHN file
            with open(file[:-7] + 'PHN') as phn:
                for line in phn:
                    # determine the right index to choose, providing `padding` extra samples on either side if possible
                    left, right, phoneme = line.split()
                    left = max(int(left) - padding, 0)
                    right = min(int(right) + padding, len(entire) - 1)

                    samples.append(entire[left:right])
                    phonemes.append(phoneme)
        else:
            warnings.warn('wav file has no phn file: {}'.format(file))
    return samples, get_indices(phonemes)  # convert to indices


def get_data(dataset='train', preprocessor=None, batch_preprocess=True, TIMIT_root='TIMIT/TIMIT/',
             use_cache=True, y_type='categorical', padding=0):
    """Return the train, val, or test set from the TIMIT directory.

    If batch_preprocess is set, the preprocessor must accept a list of data points (audio samples) and a list of
    corresponding labels (phoneme strings). Otherwise, it must accept a single data point and its corresponding
    label (phoneme string). In either case, it should return preprocessed versions of both inputs.

    The train and val sets are differentiated by using the same random seed for splitting with sklearn's
    train_test_split function.

    Args:
        dataset (str): specifies the requested dataset; one of {'train', 'val', 'test', 'toy'}.
        preprocessor (callable): preprocessing function to be applied to data. Call signature must allow (x, y) where
                                 x is a single np.ndarray of audio and y is a label (str). If batch_preprocess is True,
                                 preprocessor is called on X, y where X is a np.ndarray of all the audio and y is a list
                                 of labels.
        batch_preprocess (bool): if True, preprocessor is called on the entire dataset at once. Otherwise, preprocessor
                                 is called on a single data point and label at a time.
        TIMIT_root (str): specifies the root data directory of the TIMIT corpus. Should contain subdirectories 'TRAIN'
                          and 'TEST'.
        use_cache (bool): if True, reuses preprocessed data cached in TIMIT_root/cache if available. If False, recreates
                          dataset and caches it in that location.
        y_type (str): the type of label set to return; one of {'categorical', 'one-hot'}.
        padding (int): the number of audio samples to provide on either side of the phoneme, where available. Default is
                       no padding (0).

    Returns:
        list(np.ndarray): audio data, preprocessed as specified.
        list(str) or list(): phoneme types corresponding to the audio data.
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
    pickle_path = path.join(TIMIT_root, 'cache/{}/{}/{}.pkl'.format(dataset.lower(), fn_name, padding))
    makedirs(path.join(TIMIT_root, 'cache/{}/{}'.format(dataset.lower(), fn_name)), exist_ok=True)

    # load data from either cache or directory
    if use_cache and path.isfile(pickle_path):  # cache exists
        print('Loading {}/{}/{} set from cache...'.format(dataset.lower(), fn_name, padding), end='', flush=True)
        with open(pickle_path, 'rb') as infile:
            X, y = pickle.load(infile)
        print(' done.')
    else:  # not cached
        print('Loading {} set from files...'.format(dataset.lower()), end='', flush=True)
        # load from files
        if dataset.lower() == 'toy':
            X, y = _load_from_dir(set_root, padding=padding, max_files=100)
        else:
            X, y = _load_from_dir(set_root, padding=padding)
        print(' done.')

        # get just train set or just val set if necessary
        if dataset.lower() == 'train':
            X, _, y, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        elif dataset.lower().startswith('val'):
            _, X, _, y = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # apply preprocessor
        if preprocessor:
            print('Applying preprocessor "{}"...'.format(fn_name), end='', flush=True)
            if batch_preprocess:
                X, y = preprocessor(X, y)
            else:
                X, y = zip(*(preprocessor(x, wai) for x, wai in zip(X, y)))
                X, y = list(X), list(y)
            print(' done.')

        # cache the dataset for future use
        print('Saving {}/{}/{} set to cache...'.format(dataset.lower(), fn_name, padding), end='', flush=True)
        with open(pickle_path, 'wb+') as outfile:
            pickle.dump((X, y), outfile)
        print(' done.')

    # convert to one-hot if necessary
    if y_type.lower() == 'one-hot':
        y = to_onehot(y)

    return X, y


def test_TIMIT_phones():
    """Test get_data using default parameters."""
    X, y = get_data()
    print("running test_TIMIT_phones()")
    print('Object lengths are:', len(X), len(y))
    print('Shapes of first elements are:', X[0].shape, y[0].shape)
