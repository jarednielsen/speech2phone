r"""Preprocessing module for TIMIT data. Defines functions for loading TIMIT data, as well as various preprocessing
methods which can be applied by Preprocess.get_data.

Run this command to convert the LDC sphere files to .wav:

    find . -name '*.WAV' -exec sph2pipe -f wav {} {}.wav \;

sph2pipe is available online from the LDC.

Kyle Roth. 2019-02-05.
"""


from os import path, mkdir
from glob import iglob as glob
import warnings
import pickle
import inspect

import librosa
import numpy as np
from scipy import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# mapping from phones to integers, to be used consistently with every dataset
phones = np.array([
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em',
    'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm',
    'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w',
    'y', 'z', 'zh'
])

phone_to_idx = {
    phone: i for i, phone in enumerate(phones)
}


class Preprocess:
    """Static class for encapsulation of the get_data function."""

    rate = 16000
    mel_ = librosa.filters.mel(rate, 2048, n_mels=80)  # linear transformation, (80,1025)

    @staticmethod
    def _get_TIMIT_set_path(TIMIT_root, dataset):
        if dataset.lower() in {'train', 'val', 'toy'}:
            return path.join(TIMIT_root, 'TRAIN')
        if dataset.lower() == 'test':
            warnings.warn('loading test data; only use test data to demonstrate final results')
            return path.join(TIMIT_root, 'TEST')

        raise ValueError('dataset must be specified as one of (\'train\', \'val\', \'test\', \'toy\')')

    @staticmethod
    def get_phones(indices):
        """Take a vector of indices, and return their respective phonemes.

        Args:
            indices (iterable(int)): vector of indices
        Returns:
            np.ndarray(str): vector of phones
        """
        return phones[indices]

    @staticmethod
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

    @staticmethod
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
            y = Preprocess.get_indices(np.copy(y))
        # encode
        out[np.arange(len(y)), y] = 1
        return out

    @staticmethod
    def _load_from_dir(directory, max_files=None):
        """Load the dataset from the specified directory.

        Warn if a WAV file is encountered without a corresponding PHN file. See module docstring for instruction to
        convert from 'NIST' format to .wav.

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
                        left, right, phoneme = line.split()
                        samples.append(entire[int(left):int(right)])
                        phonemes.append(phoneme)
            else:
                warnings.warn('wav file has no phn file: {}'.format(file))
        return samples, Preprocess.get_indices(phonemes)  # convert to indices

    @staticmethod
    def get_data(dataset='train', preprocessor=None, batch_preprocess=True, TIMIT_root='TIMIT/TIMIT/', use_cache=True,
                 y_type='categorical'):
        """Return the train, val, or test set from the TIMIT directory.

        If batch_preprocess is set, the preprocessor must accept a list of data points (audio samples) and a list of
        corresponding labels (phoneme strings). Otherwise, it must accept a single data point and its corresponding
        label (phoneme string). In either case, it should return preprocessed versions of both inputs.

        The train and val sets are differentiated by using the same random seed for splitting with sklearn's
        train_test_split function.

        Returns:
            list(np.ndarray): NumPy arrays of audio data, preprocessed as specified.
            list(str): Phoneme types corresponding to the audio data.
        """
        if y_type.lower() not in ('categorical', 'one-hot'):
            raise ValueError('y_type must be one of (\'categorical\', \'one-hot\')')

        # specify the directory according to the dataset being used
        set_root = Preprocess._get_TIMIT_set_path(TIMIT_root, dataset)

        # get the name of the preprocessing function to see if it's been used before
        if preprocessor is None:
            fn_name = 'none'
        else:
            fn_name = dict(inspect.getmembers(preprocessor))['__name__']

        # ensure the caching directory is available
        if not path.isdir(path.join(TIMIT_root, 'cache/{}'.format(dataset.lower()))):
            mkdir(path.join(TIMIT_root, 'cache/{}'.format(dataset.lower())))
        pickle_path = path.join(TIMIT_root, 'cache/{}/{}.pkl'.format(dataset.lower(), fn_name))

        # load data from either cache or directory
        if use_cache and path.isfile(pickle_path):  # cache exists
            print('Loading {}/{} set from cache...'.format(dataset.lower(), fn_name), end='', flush=True)
            with open(pickle_path, 'rb') as infile:
                X, y = pickle.load(infile)
            print(' done.')
        else:  # not cached
            print('Loading {} set from files...'.format(dataset.lower()), end='', flush=True)
            # load from files
            if dataset.lower() == 'toy':
                X, y = Preprocess._load_from_dir(set_root, max_files=100)
            else:
                X, y = Preprocess._load_from_dir(set_root)
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
            print('Saving {}/{} set to cache...'.format(dataset.lower(), fn_name), end='', flush=True)
            with open(pickle_path, 'wb+') as outfile:
                pickle.dump((X, y), outfile)
            print(' done.')

        # convert to one-hot if necessary
        if y_type.lower() == 'one-hot':
            y = Preprocess.to_onehot(y)

        return X, y

    @staticmethod
    def mel(X, y):
        """Mel spectrogram preprocessing."""

        out = []
        for x in X:
            spectrum = np.log(np.abs(fft(x))[:len(x)//2])
            spectrum = signal.resample(spectrum, 1025)
            x_mel = np.dot(Preprocess.mel_, spectrum)
            out.append(x_mel)
        X = np.array(out)
        y = np.array(y)

        # remove NaNs
        is_nan = np.isnan(X).any(axis=1)
        X, y = X[~is_nan], y[~is_nan]

        return X, y


def test_preprocess():
    """Test Preprocess.get_data using default parameters."""
    result = Preprocess.get_data()
    print("running test_preprocess(); result is {}".format(result))
