import numpy as np
import librosa as lr

from matplotlib import pyplot as plt

def zeropad(x, length):
    """ Zeropads last dimension to have size = length
    
    Arguments:
        x {ndarray} -- [description]
        length {int} -- [description]
    
    Returns:
        ndarray -- zeropadded x
    """
    x_len = x.shape[-1]
    if length < x_len:
        return x[..., 0:length]
    else:
        shape = list(x.shape[0:-1])
        shape.append(length)
        x_padded = np.zeros(shape, dtype=x.dtype)
        x_padded[..., 0:x_len] = x
        
    return x_padded

def filter_lowpass(x, cutoff):
    

def filter(x, cutoff):
    return None

def spectrogram(x):
    x_float = x.astype(np.float32)

    s = np.abs(lr.stft(x_float, n_fft=256, hop_length=64, window='hann'))
    

    #plt.imshow(s.T, cmap='BuPu')
    #plt.show()
    
    return s.astype(x.dtype)


#def inverse_spectrogram(x):
#    return lr.istft(x, n_fft=512, hop_length=256, window='hann')