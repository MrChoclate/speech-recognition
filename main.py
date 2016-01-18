import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import math

from statsmodels.tsa.ar_model import AR

YES_FILES = [
    'data/oui{}.wav'.format(i) for i in range(1, 5)
]

NO_FILES = [
    'data/non{}.wav'.format(i) for i in range(5, 10)
]

NUMBERS = [
    'data/chiffre{}.wav'.format(i) for i in range(0, 10)
]

def read(filename):
    rate, data = wavfile.read(filename)
    return rate, data

def get_lpc(trame):
    ar_mod = AR(trame)
    ar_res = ar_mod.fit(20)
    return ar_res.params

def display_ar(filename):
    rate, data = read(filename)
    plt.subplot(211)
    plt.plot(data)
    ars = get_lpc(np.array(data, dtype='int64'))
    print(ars)
    data_ar = [
        sum(- ars[k] * data[n - k - 1] for k in range(len(ars)) if n - k > 0)
        for n in range(len(data))]
    plt.subplot(212)
    plt.plot(data_ar)
    plt.show()

def display_sync(filename):
    rate, data = read(filename)
    plt.subplot(211)
    plt.plot(data)
    data_sync = apply_threshold(data, threshold=10000000)
    plt.subplot(212)
    plt.plot(data_sync)
    plt.show()

def get_cepstrum(trame):
    """Returns the cepstrum coefficient of a trame"""
    Z = fft(trame)
    S = np.log(np.abs(Z))
    c = ifft(S)
    return c

def apply_threshold(data, threshold=10000000):
    i_first = next(i for i, value in enumerate(data) if value ** 2 > threshold)
    i_last = len(data) - next(i for i, value in enumerate(reversed(data))
                                if value ** 2 > threshold)
    return data[i_first:i_last]


def synchronize(signal1, signal2, threshold=10000000):
    signal1 = signal1[0], apply_threshold(signal1[1], threshold)
    signal2 = signal2[0], apply_threshold(signal2[1], threshold)
    return signal1, signal2

def split_signal(signal, dt=0.023):
    """Split the signal into frames of dt length. dt is in second"""

    rate, data = signal
    step = int(rate * dt)
    trames = [
        data[i * step:(i + 1) * step]
        for i in range(math.ceil(len(data) / step))
    ]
    return trames[:-1]

def euclidian_distance(signal1, signal2, method="cepstrum"):
    #signal1, signal2 = synchronize(signal1, signal2)
    trames1, trames2 = split_signal(signal1), split_signal(signal2)

    getter = {"cepstrum": get_cepstrum, "lpc": get_lpc}
    acoustic_vectors1 = [getter[method](trame) for trame in trames1]
    acoustic_vectors2 = [getter[method](trame) for trame in trames2]

    return sum(
        np.abs(coeff1 - coeff2) ** 2
        for v1, v2 in zip(acoustic_vectors1, acoustic_vectors2)
        for coeff1, coeff2 in zip(v1[:4], v2[:4])
    )

def dynamic_time_wraping(signal1, signal2, method="cepstrum", r=2):
    signal1, signal2 = synchronize(signal1, signal2)
    trames1, trames2 = split_signal(signal1), split_signal(signal2)

    getter = {"cepstrum": get_cepstrum, "lpc": get_lpc}
    acoustic_vectors1 = [getter[method](trame) for trame in trames1]
    acoustic_vectors2 = [getter[method](trame) for trame in trames2]

    vec1, vec2 = acoustic_vectors1, acoustic_vectors2
    return sum(
        min(sum(
            np.abs(coeff1 - coeff2) ** 2
            for coeff1, coeff2 in zip(vec1[i], vec2[i + k])
        ) for k in range(- r, r + 1) if 0 <= i + k < min(len(vec2), len(vec1)))
        for i in range(min(len(vec1), len(vec2)))
    )

def guess_yes_no(filename, method="cepstrum"):
    signal = read(filename)
    yes_signals = [read(filename) for filename in YES_FILES]
    no_signals = [read(filename) for filename in NO_FILES]

    yes_d = [euclidian_distance(signal, yes, method=method) for yes in yes_signals]
    no_d = [euclidian_distance(signal, no, method=method) for no in no_signals]

    print(yes_d, no_d)

    if min(yes_d) < min(no_d):
        return "YES"
    else:
        return "NO"

def guess_yes_no_dtw(filename, method="cepstrum"):
    signal = read(filename)
    yes_signals = [read(filename) for filename in YES_FILES]
    no_signals = [read(filename) for filename in NO_FILES]

    yes_d = [dynamic_time_wraping(signal, yes, method=method) for yes in yes_signals]
    no_d = [dynamic_time_wraping(signal, no, method=method) for no in no_signals]

    print(yes_d, no_d)

    d1, d2 = tuple(sorted(yes_d + no_d)[:2])
    if abs(d1 ** 2 - d2 ** 2) > 500000:
        return "UNKNOWN"

    if min(yes_d) < min(no_d):
        return "YES"
    else:
        return "NO"

def guess_number(filename, method="cepstrum"):
    index = 0
    signal = read(filename)
    number = [read(filename) for filename in NUMBERS]

    num_d = [euclidian_distance(signal, n, method=method) for n in number]
    print(num_d)

    index = num_d.index(min(num_d))
    in_letters = ["zero", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine"]

    return in_letters[index]

if __name__ == '__main__':

    print(guess_yes_no("data/nong5.wav", method='lpc'))
    print(guess_yes_no("data/nong6.wav", method='lpc'))
    print(guess_yes_no("data/ouig0.wav", method='lpc'))
    print(guess_yes_no("data/ouig1.wav", method='lpc'))
    print(guess_yes_no("data/yo.wav", method='lpc'))


    print(guess_number("data/candidat0.wav", method='lpc')) # 0
    print(guess_number("data/candidat1.wav", method='lpc')) # 1
    print(guess_number("data/candidat2.wav", method='lpc')) # 2
    print(guess_number("data/candidat3.wav", method='lpc')) # 3

    print(guess_yes_no("data/nong5.wav"))
    print(guess_yes_no("data/nong6.wav"))
    print(guess_yes_no("data/ouig0.wav"))
    print(guess_yes_no("data/ouig1.wav"))

    print(guess_number("data/candidat0.wav")) # 0
    print(guess_number("data/candidat1.wav")) # 1
    print(guess_number("data/candidat2.wav")) # 2
    print(guess_number("data/candidat3.wav")) # 3


    # print(guess_yes_no_dtw("data/yo.wav", method='lpc'))
    # display_ar("data/ouig0.wav")
    # display_sync("data/ouig0.wav")
