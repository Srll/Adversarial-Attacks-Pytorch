from scipy.io import wavfile
import numpy as np


def get_audio_speech():
    fs, x = wavfile.read("00f0204f_nohash_0.wav")
    x = x / np.max(x)
    return fs, x

def get_audio_sin():
    n = np.arange(0,8000)
    fs = 44100
    x = np.sin(1000* 2*np.pi*n/fs)
    #x += 0.9*np.sin(11100* 2*np.pi*n/fs)
    
    x += 0.2*np.sin(2300* 2*np.pi*n/fs)
    x += 0.1*np.sin(2100* 2*np.pi*n/fs)
    x += 0.1*np.sin(4000* 2*np.pi*n/fs)
    x += 0.1*np.sin(4300* 2*np.pi*n/fs)
    x += 0.1*np.sin(7000* 2*np.pi*n/fs)
    x += 0.1*np.sin(7100* 2*np.pi*n/fs)
    x += 0.1*np.sin(7200* 2*np.pi*n/fs)
    x += 0.1*np.sin(7300* 2*np.pi*n/fs)
    x += 0.4*np.sin(11300* 2*np.pi*n/fs)
    
    x = x / np.max(x)
    return fs, x