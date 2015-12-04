import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
from spectrum import aryule
import math


YES_FILES = [
    'data/yesno/oui{}.wav'.format(i) for i in range(1, 6)
]

NO_FILES = [
    'data/yesno/non{}.wav'.format(i) for i in range(1, 6)
]


def read(filename):
    rate, data = wavfile.read(filename)
    return rate, data

def get_lpc(trame):
    ar, p, k = aryule(trame, len(trame) - 1)
    return ar

def get_cepstrum(trame):
    """Returns the cepstrum coefficient of a trame"""
    Z = fft(trame)
    S = np.log(np.abs(Z))
    c = ifft(S)
    return c

def apply_threshold(data, threshold=100000):
    i_first = next(i for i, value in enumerate(data) if value ** 2 > threshold)
    i_last = len(data) - next(i for i, value in enumerate(reversed(data))
                                if value ** 2 > threshold)
    return np.array(data[i_first:i_last], dtype='int64')


def synchronize(signal1, signal2, threshold=100000):
    signal1 = signal1[0], apply_threshold(signal1[1], threshold)
    signal2 = signal2[0], apply_threshold(signal2[1], threshold)
    return signal1, signal2

def split_signal(signal, dt=0.03):
    """Split the signal into frames of dt length. dt is in second"""

    rate, data = signal
    step = int(rate * dt)
    trames = [
        data[i * step:(i + 1) * step]
        for i in range(math.ceil(len(data) / step))
    ]
    return trames

def euclidian_distance(signal1, signal2, method="cepstrum"):
    signal1, signal2 = synchronize(signal1, signal2)
    trames1, trames2 = split_signal(signal1), split_signal(signal2)

    getter = {"cepstrum": get_cepstrum, "lpc": get_lpc}
    acoustic_vectors1 = [getter[method](trame) for trame in trames1]
    acoustic_vectors2 = [getter[method](trame) for trame in trames2]

    return sum(
        np.abs(coeff1 - coeff2) ** 2
        for v1, v2 in zip(acoustic_vectors1, acoustic_vectors2)
        for coeff1, coeff2 in zip(v1, v2)
    )

def guess_yes_no(filename, method="cepstrum"):
    signal = read(filename)
    yes_signals = [read(filename) for filename in YES_FILES]
    no_signals = [read(filename) for filename in NO_FILES]

    yes_d = [euclidian_distance(signal, yes, method=method) for yes in yes_signals]
    no_d = [euclidian_distance(signal, no, method=method) for no in no_signals]

    print(yes_d, no_d)

    if sum(yes_d) < sum(no_d):
        return "YES"
    else:
        return "NO"


if __name__ == '__main__':
    print(guess_yes_no("data/yesno/nont0.wav", method='lpc'))
