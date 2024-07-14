"""
preprocessing

a module to process the data
"""
import numpy as np
from scipy.fft import fft, fftfreq
from statistics import LinearGraphic
import matplotlib.pyplot as plt

def BlocksToSignal(blocks):
    """
    returns a signal-representation for the blocks
    """
    freq = {}
    blocks_ids = {}
    blocks_pos = 0
    for block in blocks:
        if not tuple(block) in freq.keys():
            freq[tuple(block)] = 1
            blocks_ids[blocks_pos] = tuple(block)
            blocks_pos += 1
            pass
        else:
            freq[tuple(block)] += 1
            pass
        pass
    return [freq[key] for key in freq.keys()]

def VisualizeFrequency(signal,period=1/800,**kwargs):
    save = False
    show = True
    normalized = False
    title = 'fig'
    if 'save' in kwargs.keys():
        save = kwargs['save']
        pass
    if 'title' in kwargs.keys():
        title = kwargs['title']
        pass
    if 'show' in kwargs.keys():
        show = kwargs['show']
        pass
    if 'normalize' in kwargs.keys():
        normalized = kwargs['normalize']
        pass
    Y = fft(signal)
    if normalized:
        norm = np.abs(Y).max()
        Y = Y / norm
        pass
    xf = fftfreq(len(signal),period)[:len(signal) // 2]
    plt.figure()
    plt.plot(xf,np.abs(Y[0:len(signal) // 2]))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Value')
    plt.grid(True)
    if save:
        plt.savefig(title)
        pass
    if show:
        plt.show()
        pass
    pass

def MatrixFFT(matrix,t=1,fs=1000,**kwargs):
    """
    apply the FFT to the matrix
    """
    save = False
    show = True
    normalized = False
    title = 'fig'
    if 'save' in kwargs.keys():
        save = kwargs['save']
        pass
    if 'title' in kwargs.keys():
        title = kwargs['title']
        pass
    if 'show' in kwargs.keys():
        show = kwargs['show']
        pass
    if 'normalize' in kwargs.keys():
        normalized = kwargs['normalize']
        pass
    X = np.fft.fft2(matrix)
    if normalized:
        norm = np.abs(X).max()
        X = X / norm
        pass
    N = len(X)
    freqs = np.linspace(0,fs,N//2)
    LinearGraphic(freqs[:N//2],abs(X[:N//2]),show,save,title)
    return X