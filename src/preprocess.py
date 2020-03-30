import librosa
import numpy as np
from scipy import signal, fftpack
import torch
import matplotlib.pyplot as plt 
# TODO add MFCC, MelSPECTROGRAM, delta calculations for MFCC, quantization, bandpass


class PreProcess():
    def __init__(self, transforms, **kwargs):
        # containing all supported preprocessing transforms and their corresponding inverse 
        
        valid_transforms = {None: (self.dummy, self.dummy),
                            'None': (self.dummy, self.dummy),
                            'stft': (self.stft, self.istft),
                            'filter': (self.filt, self.ifilt), 
                            'spectrogram':(self.spectrogram, self.ispectrogram),
                            'MFCC':(self.MFCC, self.iMFCC),
                            'mag2db':(self.mag2db, self.db2mag),
                            'mag2db2':(self.mag2db2, self.db2mag2),
                            'insert_rgb_dim': (self.push_rgb_dim, self.pop_rgb_dim),
                            'insert_data_dim': (self.push_data_dim, self.pop_data_dim),
                            'visualize': (self.plot,self.plot),
                            'dbg': (self.dbg, self.dbg),
                            'normalize': (self.normalize, self.inormalize),
                            'normalize_batch': (self.normalize_batch, self.inormalize_batch),}


        self.flag_direction = None # used for torubleshooting
        #self.kwargs = kwargs

        self.temp = 1
        # TODO make it possible to specify these through CLI
        self.kwargs =  {'coeffs_denominator' : signal.butter(6,5000,btype='low',fs=16000)[1],
                        'coeffs_numerator' : signal.butter(6,5000,btype='low',fs=16000)[0], 
                        'stft_n_fft':512,
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

    def mag2db2(self,x):
        x[x == 0] = np.finfo(float).eps
        PSD = 10 * np.log10(np.square(np.abs(x)))
        P = 96 - np.max(PSD) + PSD # should this be over one FFT or over STFT?
        #self.max = np.max(PSD, axis=1)
        self.max = np.max(PSD)
        return P

    def db2mag2(self,P):
        PSD = np.zeros_like(P)
        #for b in range(P.shape[0]):
        #    PSD[b] = P[b] - 96 + self.max[b]
        PSD = P - 96 + self.max
        x = np.sqrt(np.power(10, PSD/10))
        return x


    def mag2db(self, x):
        #TODO fix divison by 0
        x[x == 0] = np.finfo(float).eps
        return 20 * np.log10(x)
    
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
        f, t, s = signal.stft(x, nperseg=512, noverlap=256,nfft=self.kwargs.get('stft_n_fft'))
        return f,t,s

    def istft(self, s):
        _, x = signal.istft(s,nperseg=512, noverlap=256,nfft=self.kwargs.get('stft_n_fft'))
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
    
    def push_data_dim(self, x):
        dims = len(x.shape)
        x = np.squeeze(x)
        if dims == 3:
            x_new = np.zeros((x.shape[0],) + (1,) + (x.shape[1:]))
            x_new[:,0] = x
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

    def pop_data_dim(self, x):
        x_new = np.zeros((x.shape[0],) + x.shape[2:])
        x_new = x[:,0]
        return x_new



    def spectrogram(self, x):
        
        _,_,s = self.stft(x)
        self.phi = np.arctan2(s.real, s.imag)
        
        s_pow = np.sqrt(np.power(s.real, 2) + np.power(s.imag, 2 ))
        return s_pow

    def ispectrogram(self, s_pow):
        s_real = s_pow * np.sin(self.phi)
        s_imag = s_pow * np.cos(self.phi)
        s = s_real + s_imag*1.0j
        x = self.istft(s)
        return x
    

    def MFCC(self, x): # TODO
        def f_to_m(f):
            return 2595 * np.log10(1 + f/700)
        def m_to_f(m):
            return 700*(np.pow(10,m/2595) - 1)
        
        fs = 16000
        NR_BINS = 20 # 26 is a usual amount of MEL bins

        _,_,s = self.stft(x)
        self.phi = np.arctan2(s.real, s.imag) # save for reconstruction
        s_pow = np.sqrt(np.power(s.real, 2) + np.power(s.imag, 2 ))
        
        s_pow = np.swapaxes(s_pow, 1, 2)
        
        f = fs/2 * np.arange(0,s_pow.shape[2]) / s_pow.shape[2] 
        m = f_to_m(f)
        
        m_bins = NR_BINS * m / np.max(m)
        # assign to bins TODO add triangle window
        m_bins = np.floor(m_bins)
        # find all tranistions between values
        idx = np.where(m_bins[:-1] != m_bins[1:])[0]

        banks = np.zeros((s_pow.shape[:-1] + (NR_BINS,)))

        self.bins_distribution = []
        for k in range(1, idx.shape[0]-1):
            s_pow_bin_left = s_pow[..., idx[k-1]:idx[k]]
            s_pow_bin_right = s_pow[..., idx[k]:idx[k+1]]
           
            left_triangle = np.arange(1,1+idx[k]-idx[k-1]) / (idx[k]-idx[k-1])
            right_triangle = np.arange(1,1+idx[k+1]-idx[k]) / (idx[k+1]-idx[k])
            
            
            

            banks[..., k] = np.sum(s_pow_bin_left * left_triangle, axis=2) + np.sum(s_pow_bin_right * right_triangle, axis=2)

            left = s_pow_bin_left * left_triangle / np.expand_dims(banks[..., k], axis=-1)
            right = s_pow_bin_right * right_triangle / np.expand_dims(banks[..., k], axis=-1)


            self.bins_distribution.append((left, right)) # save for reconstruction

        

        banks_db = self.mag2db(banks)
        
        melFCC = fftpack.dct(banks_db, axis=2)
        melFCC = np.swapaxes(melFCC, 1, 2)
        import pdb; pdb.set_trace()

        return 

    def iMFCC(self, x):

        fs = 16000 # TODO 
        # 26 is a usual amount of MEL bins
        NR_BINS = 4

        

        melFCC = np.swapaxes(x, 1, 2)
        banks_db = fftpack.idct(melFCC, axis=2)

        banks = self.db2mag(banks_db)


        s_pow = np.zeros(banks.shape[:-1] + (129,)) # TODO not hardcode
        
        idx_low = 0
        idx_high = self.bins_distribution[0][0].shape[-1]
        
        
        for k in range(NR_BINS-2):
            print(k)
            s_pow[..., idx_low:idx_high] = self.bins_distribution[k][0] * np.expand_dims(banks[..., k], axis=-1)
            idx_low = idx_high
            idx_high += self.bins_distribution[k][1].shape[-1]
            s_pow[..., idx_low:idx_high] = self.bins_distribution[k][1] * np.expand_dims(banks[..., k], axis=-1)
            
        s_pow = np.swapaxes(s_pow, 1, 2)
        
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
        self.get_masking_threshold(x)
        
        import pdb
        pdb.set_trace()
        return x


    """
    def get_masking_threshold(self, x):
        # algorithm based on values from "Audio Watermark, A Comprehensive Foundation Using MATLAB" from 2015
        
        def quite_threshold(f):
            # output is in dB
            threshold = 3.64*np.power(f/1000, -0.8) \
                - 6.5 * np.exp(-0.6*(np.square((f/1000) - 3.3))) \
                + 1e-3 * np.power(f/1000, 4)
            return threshold
        
        def local_maximas(p):
            # returns index of all local extrema over frequency axis
            return np.stack(signal.argrelextrema(p, np.less, axis=1), axis=-1) 
            

        def hz_to_bark(f):
            b = 13 * np.arctan(0.76*f / 1000) + 3.5 * np.square(np.arctan(f/7500))
            return b
        quite_threshold
        hz_to_bark



        nfft = 512
        n = np.arange(0, nfft)
        w = np.sqrt(8/3) * 1/2 * (1 - np.cos(2*np.pi*n/nfft)) # modified hanning window (sum = 1 in power spectral domain)
        f,_,s = signal.stft(x, window=w, nperseg=nfft, fs=16000) # TODO fix correct fs
        psd = 1/2*self.mag2db(np.square(np.abs(s)))
        
        
        p = np.swapaxes(96 - np.max(psd, axis=1) + np.swapaxes(psd,0,1),0,1) # max over frequency axis
        
        p_m_idxs = local_maximas(p)

        STM = []
        for i in range(p_m_idxs.shape[0]):
            masker = True
            if 1 < p_m_idxs[i][1] < 62: # check what bark range to use
                dk_l = [-2, 2]
            elif 61 < p_m_idxs[i][1] < 127:
                dk_l = [-3,-2,2, 3]
            elif 126 < p_m_idxs[i][1] < 249:
                dk_l = [-6,-5,-4,-3,-2, 2,3,4,5,6]
            else:
                masker = False
                dk_l = []
            
            for dk in dk_l:
        
                if p[p_m_idxs[i][0],p_m_idxs[i][1],p_m_idxs[i][2]] - p[p_m_idxs[i][0],p_m_idxs[i][1]+dk,p_m_idxs[i][2]] <= 7:
                    masker = False
                    break

            if masker == True:
                STM.append(i)


            

        #for i in range(p_)
        #p_TM = 
    """









