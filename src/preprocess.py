import librosa
import numpy as np
from scipy import signal
import torch
import matplotlib.pyplot as plt 
# TODO add MFCC, MelSPECTROGRAM, delta calculations for MFCC, quantization, bandpass


class PreProcess():
    def __init__(self, transforms, **kwargs):
        # containing all supported preprocessing transforms and their corresponding inverse 
        
        valid_transforms = {'None': (self.dummy, self.dummy),
                            'stft': (self.stft, self.istft),
                            'filter': (self.filt, self.ifilt), 
                            'spectrogram':(self.spectrogram, self.ispectrogram),
                            'mag2db':(self.mag2db, self.db2mag),
                            'insert_rgb_dim': (self.push_rgb_dim, self.pop_rgb_dim),
                            'visualize': (self.plot,self.plot),
                            'dbg': (self.dbg, self.dbg),
                            'normalize': (self.normalize, self.inormalize),
                            'normalize_batch': (self.normalize_batch, self.inormalize_batch)}


        self.flag_direction = None # used for torubleshooting
        #self.kwargs = kwargs

        self.temp = 1
        # TODO make it possible to specify these through CLI
        self.kwargs =  {'coeffs_denominator' : signal.butter(6,5000,btype='low',fs=16000)[1],
                        'coeffs_numerator' : signal.butter(6,5000,btype='low',fs=16000)[0], 
                        'stft_n_fft':256,
                        'max_value':100000}


        # stores values from forward pass that are needed during the inverse pass
        self.phi = None     # stores angle of stft
        self.x_shape = None # stores size of x


        self.transforms_forward = list()
        self.transforms_inverse = list()
        for transform in transforms:
            self.transforms_forward.append(valid_transforms[transform][0]) # add all forward transforms to forward list
            self.transforms_inverse.append(valid_transforms[transform][1]) # add their corresponding "inverse" to inverse list
        self.transforms_inverse.reverse() 

    def forward(self, x):
        
        x = x.numpy()
        for t in self.transforms_forward:
            x = t(x)
        
        return torch.from_numpy(x).to(torch.float32)

    def inverse(self, x):
        x = x.numpy()
        
        for t in self.transforms_inverse:
            x = t(x)
        
        return torch.from_numpy(x).to(torch.float32)

    """def normalize(self, x):
    """
    



    def normalize_batch(self, x):
        if (len(x.shape) == 4):
            self.temp = np.max(x)
            return x / self.temp
        else:
            print("size of input doesn't allow for normalize_batch")
            return x
        

    def inormalize_batch(self, x):
        return x * self.temp

    # Preprocessing methods
    def normalize(self, x):
        x = x / self.kwargs.get('max_value')
        return x

    def inormalize(self, x):
        x = x * self.kwargs.get('max_value')
        return x

    def dummy(self, x):
        return x

    def mag2db(self, x):
        #TODO fix divison by 0
        return 20 * np.log10(x+0.0001)
    def db2mag(self, x):
        return np.power(10, x/20)

    def filt(self, x):
        #TODO change lfilter method to scipy recomended method (faster)
        y = x.copy()
        for b in range(x.shape[0]):
            y[b, :] = signal.lfilter(self.kwargs.get('coeffs_numerator'), self.kwargs.get('coeffs_denominator'), x[b,:])[:]
        return y

    def ifilt(self, x):
        #TODO change lfilter method to scipy recomended method (faster)
        y = x.copy()
        for b in range(x.shape[0]):
            y[b, :] = signal.lfilter(self.kwargs.get('coeffs_denominator'), self.kwargs.get('coeffs_numerator'), x[b,:])[:]
        return y

    def stft(self, x):
        _, _, s = signal.stft(x, nperseg=128, noverlap=64,nfft=self.kwargs.get('stft_n_fft'))
        return s

    def istft(self, s):
        _, x = signal.istft(s,nperseg=128, noverlap=64,nfft=self.kwargs.get('stft_n_fft'))
        return x
    
    def push_rgb_dim(self, x):
        dims = len(x.shape)
        x = np.squeeze(x)
        if dims == 3:
            x_new = np.zeros((x.shape[0],) + (3,) + (x.shape[1:]))
            x_new[:,0] = x / 0.3
            x_new[:,1] = x / 0.59
            x_new[:,2] = x / 0.11
        else:
            print("Not supported size for RGB conversion")
        return x_new
    
    def pop_rgb_dim(self, x):
        x_new = np.zeros((x.shape[0],) + x.shape[2:])
        if x.shape[1] == 3:
            x_new += x[:,0] * 0.3
            x_new += x[:,1] * 0.59
            x_new += x[:,2] * 0.11
        return x_new / 3

    def spectrogram(self, x):
        
        s = self.stft(x)
        self.phi = np.arctan2(s.real, s.imag)
        s_pow = np.sqrt(np.power(s.real, 2) + np.power(s.imag, 2 ))
        
        return s_pow

    def ispectrogram(self, s_pow):
        
        s_real = s_pow * np.sin(self.phi)
        s_imag = s_pow * np.cos(self.phi)
        s = s_real + s_imag*1.0j
        x = self.istft(s)
        return x
    
    def plot(self, x):
        if len(x.shape) == 3:
            x_p = x[0]
        elif len(x.shape) == 4:
            x_p = np.moveaxis(x[0], 0, -1)
            x_p = x_p / np.max(x_p)
        else:
            print("size of x doesn't allow for plot")
            return x
        plt.imshow(x_p, origin='lower')
        
        plt.ion()
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001)
        return x

    def dbg(self, x):
        import pdb
        pdb.set_trace()
        return x



    def get_masking_threshold(self, x):
        # algorithm based on values from "Audio Watermark, A Comprehensive Foundation Using MATLAB" from 2015
        
        def quite_threshold(f):
            # output is in dB
            threshold = 3.64*np.power(f/1000, -0.8) 
                        - 6.5 np.exp(-0.6*(np.square((f/1000) - 3.3)))
                        + 1e-3 * np.power(f/1000, 4)
            return threshold
        
        def local_maximas(p):
            # returns index of all local extrema over frequency axis
            return np.stack(signal.argrelextrema(p, np.less, axis=1), axis=-1) 
            

        def hz_to_bark(f):
            b = 13 * np.arctan(0.76*f / 1000) + 3.5 * np.square(np.arctan(f/7500))
            return b
        
        



        nfft = 512
        n = np.arange(0, nfft-1)
        w = np.sqrt(8/3) * 1/2 * (1 - np.cos(2*np.pi*n/nfft)) # modified hanning window (sum = 1 in power spectral domain)
        f,_,s = signal.stft(x, window=h, nperseg=nfft, fs=16000) # TODO fix correct fs
        psd = 10*np.log10((np.square(np.abs(s))))
        p = 96 - np.max(psd, axis=1) + psd # max over frequency axis
        
        p_m_idxs = local_maximas(p)

        for i in range(p_)
        p_TM = 





