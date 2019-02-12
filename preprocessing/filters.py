"""Defines preprocessing filters which can be applied by preprocessing.get_data.

Kyle Roth. 2019-02-12.
"""


import librosa
import numpy as np
from scipy import fft
from scipy.signal import resample


# used by mel function
_mel = librosa.filters.mel(16000, 2048, n_mels=80)  # linear transformation, (80,1025)


def mel(X, y):
    """Mel spectrogram preprocessing."""
    out = []
    for x in X:
        spectrum = np.log(np.abs(fft(x))[:len(x)//2])
        spectrum = resample(spectrum, 1025)
        x_mel = np.dot(_mel, spectrum)
        out.append(x_mel)
    X = np.array(out)
    y = np.array(y)

    # remove NaNs
    is_nan = np.isnan(X).any(axis=1)
    X, y = X[~is_nan], y[~is_nan]

    return X, y
