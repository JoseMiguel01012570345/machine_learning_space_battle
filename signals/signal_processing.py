"""
signal processing

a module to work with matrix as signals
"""
import numpy as np
from scipy.fft import fft
from scipy.signal import spectrogram

class SpectralFeatures:
    
    def __init__(self,signal):
        self._freq = fft(signal)
        self._centroid = spectral_centroid(self._freq)
        self._bandwith = spectral_bandwidth(self._freq)
        self._flatness = spectral_flatness(signal)
        self._contrast = spectral_contrast(signal)
        self._ones_average = np.mean(signal)
        self._average = np.sum(signal)
        pass
    
    @property
    def Centroid(self):
        return self._centroid
    
    @property
    def Bandwidth(self):
        return self._bandwith
    
    @property
    def Contrast(self):
        return self._contrast
    
    @property
    def Flatness(self):
        return self._flatness
    
    @property
    def OnesAverage(self):
        return self._ones_average
    
    @property
    def Average(self):
        return self._average
    
    pass

def matrix2canonics_coefs(matrix):
    """
    returns the coordinates of the matrix in its canonic base
    """
    x_size,y_size = matrix.shape
    coefs = np.array([0]*x_size*y_size)
    for i in range(x_size):
        for j in range(y_size):
            coefs[i*x_size + j] = matrix[i,j]
            pass
        pass
    return coefs

def spectral_centroid(freq):
    """
    returns the spectral centroid of the frequency
    """
    return np.sum(np.abs(freq[:-1]) * np.arange(len(freq[:-1]))) / np.sum(np.abs(freq[:-1]))

def spectral_bandwidth(freq):
    """
    returns the spectral bandwidth of the frequency
    """
    return np.sum(np.abs(freq[:-1])**2 * np.arange(len(freq[:-1]))**2) / (spectral_centroid(freq)**2)

def spectral_rolloff(signal,percent=0.85,fs=1.0):
    """
    return the rolloff of the signal
    percent -> float in interval (0,1)
    """
    freq,_,Sxx = spectrogram(signal,fs)
    print(freq)
    rolloff_index = np.argmax(np.cumsum(Sxx,axis=0) >= percent) + 1
    return freq[rolloff_index]

def spectral_contrast(signal,fs=1.0):
    """
    return the spectral contrast for the signal
    """
    Sxx = np.abs(spectrogram(signal,fs)[2])
    return np.mean(np.diff(np.log(Sxx)))

def spectral_flatness(signal,fs=1.0):
    """
    return the spectral flatness for the signal
    """
    Sxx = np.abs(spectrogram(signal,fs)[2])
    total_energy = np.sum(Sxx)
    mean_energy = np.mean(Sxx)
    return np.sqrt(total_energy / mean_energy)



def get_signal_features(signal):
    """
    returns the principal features of the given signal
    return type -> SpectralFeatures
    """
    return SpectralFeatures(signal)