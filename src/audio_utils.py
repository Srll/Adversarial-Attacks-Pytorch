import numpy as np
import librosa as lr
from scipy import signal
from matplotlib import pyplot as plt

<<<<<<< HEAD
def zeropad(x, target_len):
=======
def zeropad(x, length):
>>>>>>> f58cbb07312240ad2267aa9f47b37cf7f886d574
    """ if len(x)<length: Zeropads last dimension to have size = length
        else: return x[0:length]

    Arguments:
        x {ndarray} -- [description]
        length {int} -- [description]
    
    Returns:
        ndarray -- zeropadded x
    """
    x_len = x.shape[-1]
<<<<<<< HEAD
    
    if target_len < x_len:
        return x[..., 0:target_len]
        
    else:
        shape = x.shape[0:-1]
        x_padded = np.zeros(shape + (target_len,), dtype=x.dtype)
=======
    if length < x_len:
        return x[..., 0:length]
    else:
        shape = list(x.shape[0:-1])
        shape.append(length)
        x_padded = np.zeros(shape, dtype=x.dtype)
>>>>>>> f58cbb07312240ad2267aa9f47b37cf7f886d574
        x_padded[..., 0:x_len] = x
        
    return x_padded

def filter_lowpass(x, cutoff_khz, fs):
    w = 2*cutoff_khz/fs
    b, a = signal.butter(3, w)
    y = signal.filtfilt(b, a, x)
    return y

<<<<<<< HEAD



def spectrogram(x):
    x_float = x.astype(np.float32)
    s = np.abs(lr.stft(x_float, n_fft=256, hop_length=64, window='hann'))
    s_magnitude, s_phase = lr.magphase(s)

    s_magnitude_db = lr.amplitude_to_db(s_magnitude) 


=======
def spectrogram(x):
    x_float = x.astype(np.float32)
    s1 = np.abs(lr.stft(x_float, n_fft=256, hop_length=64, window='hann'))
>>>>>>> f58cbb07312240ad2267aa9f47b37cf7f886d574
    #s2 = np.abs(signal.stft(x_float, nfft=256, noverlap=256-64, window='hann'))

    #plt.imshow(s.T, cmap='BuPu')
    #plt.show()
<<<<<<< HEAD
    return s_magnitude_db.astype(x.dtype)
=======
    return s1.astype(x.dtype)
>>>>>>> f58cbb07312240ad2267aa9f47b37cf7f886d574


#def inverse_spectrogram(x):
#    return lr.istft(x, n_fft=512, hop_length=256, window='hann')