import numpy as np
from scipy import signal

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut=1, highcut=20, fs=100):
    for i in range(data.shape[0]):
        b, a = butter_bandpass(lowcut, highcut, fs)
        data[i, :] = signal.lfilter(b, a, data[i, :])
    return data

def normalize(data):
    return (data - np.mean(data, axis=0, keepdims=True)) / (np.std(data, axis=0, keepdims=True) + 1e-5)

def re_com_mode(data):
    for i in range(data.shape[1]):
        data[:, i] -= np.mean(data[:, i])
    return data