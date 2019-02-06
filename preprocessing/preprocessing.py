"""Preprocessing module for TIMIT data. Defines functions for loading TIMIT data, as well as various preprocessing
methods which can be applied by Preprocess.get_data.

Run this command to convert the LDC sphere files to .wav:

    find . -name '*.WAV' -exec sph2pipe -f wav {} {}.wav \;

sph2pipe is available online from the LDC.

Kyle Roth. 2019-02-05.
"""


from os import path
from glob import iglob as glob
import librosa
import numpy as np
import scipy as sp
from scipy import signal
import warnings

from scipy.io import wavfile
from sklearn.model_selection import train_test_split

class Preprocess:
    """Static class for encapsulation of the get_data function."""
    
    rate = 16000
    mel_ = librosa.filters.mel(rate, 2048, n_mels=80) # linear transformation, (80,1025)

    @staticmethod
    def _load_from_dir(directory, max_files=None):
        """Load the dataset from the specified directory.

        Warn if a WAV file is encountered without a corresponding PHN file. See module docstring for instruction to
        convert from 'NIST' format to .wav.

        Returns:
            list(np.ndarray): NumPy arrays of audio data.
            list(str): Phoneme types corresponding to the audio data.
        """
        samples = []
        phonemes = []

        file_list = list(glob(path.join(directory, '**/*.WAV.wav'), recursive=True))
        print("num_files: {}".format(len(file_list)))
        if max_files is not None:
            file_list = file_list[:max_files]

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
        return samples, phonemes

    @staticmethod
    def get_data(dataset='train', preprocessor=None, batch_preprocess=True, TIMIT_root='TIMIT/TIMIT/', max_files=None):
        """Return the train, validation, or test set from the TIMIT directory.

        If batch_preprocess is set, the preprocessor must accept a list of data points (audio samples) and a list of
        corresponding labels (phoneme strings). Otherwise, it must accept a single data point and its corresponding
        label (phoneme string). In either case, it should return preprocessed versions of both inputs.

        The train and validation sets are differentiated by using the same random seed for splitting with sklearn's
        train_test_split function.

        Returns:
            list(np.ndarray): NumPy arrays of audio data, preprocessed as specified.
            list(str): Phoneme types corresponding to the audio data.
        """
        # specify the directory according to the dataset being used
        if dataset.lower() in {'train', 'val', 'validation'}:
            TIMIT_root = path.join(TIMIT_root, 'TRAIN')
        elif dataset.lower() == 'test':
            warnings.warn('loading test data; only use test data to demonstrate final results')
            TIMIT_root = path.join(TIMIT_root, 'TEST')
        else:
            raise ValueError('dataset must be specified as one of (\'train\', \'val\', \'test\')')

        # load data from the directory
        X, y = Preprocess._load_from_dir(TIMIT_root, max_files=max_files)

        # get just train set or just val set if necessary
        if dataset.lower() == 'train':
            X, _, y, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        elif dataset.lower().startswith('val'):
            _, X, _, y = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # apply preprocessor
        if preprocessor:
            if batch_preprocess:
                X, y = preprocessor(X, y)
            else:
                X, y = zip(*(preprocessor(x, wai) for x, wai in zip(X, y)))
                X, y = list(X), list(y)

        return X, y
    
    @staticmethod
    def mel(X, y):
        """Mel spectrogram preprocessing.
        """
        spectrum = np.log(np.abs(sp.fft(X))[:len(X)//2])
        spectrum = signal.resample(spectrum, 1025)
        X_mel = np.dot(Preprocess.mel_, spectrum)
        return X_mel, y
        


def test_preprocess():
    """Test Preprocess.get_data using default parameters."""
    result = Preprocess.get_data()
    print("running test_preprocess(); result is {}".format(result))
