import numpy as np
import librosa as lr
from scipy import signal
from matplotlib import pyplot as plt


def resample_to_44100(x, Fs):
    nr_of_samples = int(x.shape[0] * (44100/Fs))
    x = signal.resample(x, nr_of_samples)
    return x

def zeropad(x, target_len):
    """ if len(x)<length: Zeropads last dimension to have size = length
        else: return x[0:length]

    Arguments:
        x {ndarray} -- [description]
        length {int} -- [description]
    
    Returns:
        ndarray -- zeropadded x
    """
    x_len = x.shape[-1]
    
    if target_len < x_len:
        return x[..., 0:target_len]
        
    else:
        shape = x.shape[0:-1]
        x_padded = np.zeros(shape + (target_len,), dtype=x.dtype)
        x_padded[..., 0:x_len] = x
        
    return x_padded

def filter_lowpass(x, cutoff_khz, fs):
    w = 2*cutoff_khz/fs
    b, a = signal.butter(3, w)
    y = signal.filtfilt(b, a, x)
    return y




def spectrogram(x):
    x_float = x.astype(np.float32)
    s = np.abs(lr.stft(x_float, n_fft=256, hop_length=64, window='hann'))
    s_magnitude, s_phase = lr.magphase(s)

    s_magnitude_db = lr.amplitude_to_db(s_magnitude) 


    #s2 = np.abs(signal.stft(x_float, nfft=256, noverlap=256-64, window='hann'))

    #plt.imshow(s.T, cmap='BuPu')
    #plt.show()
    return s_magnitude_db.astype(x.dtype)


#def inverse_spectrogram(x):
#    return lr.istft(x, n_fft=512, hop_length=256, window='hann')