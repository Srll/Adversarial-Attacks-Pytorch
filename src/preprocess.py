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
                            'mag2db96':(self.mag2db96, self.db2mag96),
                            'insert_rgb_dim': (self.push_rgb_dim, self.pop_rgb_dim),
                            'insert_data_dim': (self.push_data_dim, self.pop_data_dim),
                            'visualize': (self.plot,self.plot),
                            'dbg': (self.dbg, self.dbg),
                            'normalize': (self.normalize, self.inormalize),
                            'normalize_batch': (self.normalize_batch, self.inormalize_batch),
                            'resample_to_44100': (self.resample_to_44100, self.iresample_to_44100)}


        self.flag_direction = None # used for torubleshooting
        #self.kwargs = kwargs

        self.fs = 16000
        
        # TODO make it possible to specify these through CLI
        self.kwargs =  {'coeffs_denominator' : signal.butter(6,5000,btype='low',fs=16000)[1],
                        'coeffs_numerator' : signal.butter(6,5000,btype='low',fs=16000)[0], 
                        'max_value':100000}
        self.stft_n_fft = 512
                        


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
        if torch.is_tensor(x):
            x = x.numpy()
        
        for t in self.transforms_forward:
            x = t(x)
        
        return torch.from_numpy(x).to(torch.float32)

    def inverse(self, x):
        if torch.is_tensor(x):
            x = x.numpy()
        
        for t in self.transforms_inverse:
            x = t(x)
        x = (x*np.power(2,16)).astype(int)/np.power(2,16) # quantize to 16 bit
        
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
        self.mean = np.mean(x, axis=1)
        self.std = np.std(x, axis=1)
        x = ((x.T - self.mean) / self.std).T
        
        return x

    def inormalize(self, x):
        x = (x.T * self.std + self.mean).T
        #x = x * self.kwargs.get('max_value')
        return x

    def dummy(self, x):
        return x

    def mag2db96(self,x):
        x[x == 0] = np.finfo(float).eps
        PSD = 10 * np.log10(np.square(np.abs(x)))
        self.max = np.max(PSD,axis=(1,3)).T
        P = np.swapaxes(96 + np.swapaxes(PSD, 0, -1) - self.max, 0, -1)
        return P

    def db2mag96(self,P):
        PSD = np.swapaxes(np.swapaxes(P,0,-1) - 96 + self.max,0,-1)
        x = np.sqrt(np.power(10, PSD/10))
        return x

    def resample_to_44100(self, x):
        self.original_size = x.shape[1]
        nr_of_samples = int(x.shape[1] * (44100/self.fs))
        x = signal.resample(x, nr_of_samples, axis=1)
        return x

    def iresample_to_44100(self, x):
        x = signal.resample(x,self.original_size, axis=1)
        return x

    
    def mag2db(self, x):
        
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
        f, t, s = signal.stft(x, nperseg=self.stft_n_fft, noverlap=self.stft_n_fft/2,nfft=self.stft_n_fft)
        return f,t,s

    def istft(self, s):
        _, x = signal.istft(s,nperseg=self.stft_n_fft, noverlap=self.stft_n_fft/2,nfft=self.stft_n_fft)
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



    def spectrogram_mel(self, x):
        # FIX
        _,_,s = self.stft(x)
        self.phi = np.arctan2(s.real, s.imag)
        
        s_pow = np.power(s.real, 2) + np.power(s.imag, 2)
        return s_pow

    def spectrogram(self, x):
        _,_,s = self.stft(x)
        self.phi = np.arctan2(s.real, s.imag)
        
        s_abs = np.abs(s)
        return s_abs

    def ispectrogram(self, s_abs):
        s_real = s_abs * np.sin(self.phi)
        s_imag = s_abs * np.cos(self.phi)
        s = s_real + s_imag*1.0j
        x = self.istft(s)
        return x
    


    
    def MFCC(self, x): # TODO
        def f_to_m(f):
            return 2595 * np.log10(1 + f/700)
        def m_to_f(m):
            return 700*(np.power(10,m/2595) - 1)
        
        fs = 16000
        
        NR_BINS = 26

        x_pre_emphasis = x[:,1:] - 0.9*x[:,:-1]

        #_,_,s = self.stft(x)
        s = (self.spectrogram(x_pre_emphasis))**2
        #s = self.spectrogram(x_pre_emphasis)
        s = np.swapaxes(s, 1, 2)
        f_res = s.shape[2]
        

        mel_max = f_to_m(fs/2)
        mel_min = 0
        mel_points = np.linspace(mel_min, mel_max, NR_BINS + 2)
        hz_points = m_to_f(mel_points) # center point of each filter defined in Hz


        bins = np.floor((f_res + 1) * hz_points / fs)

        fbank = np.zeros((NR_BINS, f_res))
        for m in range(1, NR_BINS + 1):
            f_m_minus = int(bins[m - 1])   # left
            f_m = int(bins[m])             # center
            f_m_plus = int(bins[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
        
        
        s_mel = np.dot(s, fbank.T)
        
        s_mel_log = self.mag2db2(s_mel)
        mfcc = fftpack.dct(s_mel_log, axis=2, norm='ortho')
        mfcc = np.swapaxes(mfcc,1,2)
        
        #temp = np.swapaxes(temp,0,2)

        imx = librosa.feature.mfcc(y=x_pre_emphasis[0], sr=fs, hop_length=256, norm='ortho', n_mfcc=26)
        #plt.imshow(imx)
        #plt.show()

        #plt.imshow(mfcc[0])
        #plt.show()
        
        
        mfcc_1 = np.zeros(mfcc.shape)
        mfcc_2 = np.zeros(mfcc.shape)
        mfcc_1[:,:,1:] = mfcc[... ,0:-1] - mfcc[... ,1:]
        mfcc_2[:,:,1:] = mfcc_1[... ,0:-1] - mfcc_1[... ,1:]
        mfcc = np.concatenate((mfcc, mfcc_1,mfcc_2), axis=1)

        return mfcc
    

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
            x_p = x[-1]
        elif len(x.shape) == 4:
            x_p = np.moveaxis(x[0], 0, -1)
            x_p = x_p / np.max(x_p)
        else:
            print("size of x doesn't allow for plot")
            return x
        plt.imshow(x_p, origin='lower')
        
        plt.ion()
        plt.draw()
        #plt.show(block=False)
        plt.show(block=True)
        plt.pause(0.001)
        return x

    def dbg(self, x):
        
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









