import preprocess
import numpy as np
import torch
from matplotlib import pyplot as plt


def preprocess_test():
    fs = 8000
    t = np.arange(1,fs*2)
    x = np.random.random((16,t.shape[0])) + np.sin(100*t*2*np.pi/fs) # batchdim, datadim
    x = torch.from_numpy(x)
    
    def stft_test():
        p = preprocess.PreProcess(['stft'])
        s = p.forward(x)
        x_hat = p.inverse(s)
        print(np.max(x_hat - x))

    def spectrogram_test():
        p = preprocess.PreProcess(['spectrogram'])
        s = p.forward(x)
        x_hat = p.inverse(s)
        print(np.max(x_hat - x))

    def MFCC_test():
        p_ = preprocess.PreProcess(['MFCC'])
        s_ = p_.forward(x)
        
        x_hat = p_.inverse(s_)
        print(x_hat)
        import pdb; pdb.set_trace()
        print(torch.max(x_hat[0,0:1000] - x[0,0:1000]))
    
    MFCC_test()
preprocess_test()
