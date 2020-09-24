import numpy as np
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

