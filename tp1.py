import random
import math

import numpy
import matplotlib.pyplot as plt

def white_noise(K):
    return numpy.array([random.random() - 0.5 for _ in range(K)])

def bruit_blanc(K, m, v):
    return math.sqrt(12 * v) * white_noise(K) + m

def mean(x):
    return sum(x) / len(x)

def var(x):
    m = mean(x)
    return sum((val  - m) ** 2 for val in x) / len(x)

def autocov(data, tau):
    if tau == 0:
        return var(data)

    res = 0
    m = mean(data)
    for i, x in enumerate(data[:-tau]):
        res += (x - m) * (data[i + tau] - m)
    return res / len(data - tau)

if __name__ == '__main__':
    data_bruit_blanc = bruit_blanc(1000, 0, 100)
    plt.subplot(211)
    plt.plot(data_bruit_blanc)
    plt.subplot(212)
    plt.plot([autocov(data_bruit_blanc, i) for i in range(100)])
    plt.show()

    print(mean(data_bruit_blanc))
    print(var(data_bruit_blanc))
