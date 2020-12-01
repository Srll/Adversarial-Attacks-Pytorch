import librosa
import numpy as np
from scipy import signal, fftpack
import torch
import matplotlib.pyplot as plt 
# TODO add MFCC, MelSPECTROGRAM, delta calculations for MFCC, quantization, bandpass


class PreProcess():
    def __init__(self, transforms, **kwargs):
        # containing all supported preprocessing transforms and their corresponding inverse.
        self.base_type = torch.float32
        valid_transforms = {None: (self.dummy, self.dummy),
                            'None': (self.dummy, self.dummy),
                            'stft': (self.stft, self.istft),
                            'spectrogram':(self.spectrogram, self.ispectrogram),
                            'spectrogram32':(self.spectrogram32, self.ispectrogram32),
                            'MFCC':(self.MFCC, self.iMFCC),
                            'mag2db':(self.mag2db, self.db2mag),
                            'mag2db96':(self.mag2db96, self.db2mag96),
                            'insert_rgb_dim': (self.push_rgb_dim, self.pop_rgb_dim),
                            'insert_data_dim': (self.push_data_dim, self.pop_data_dim),
                            'visualize': (self.plot,self.plot),
                            'dbg':(self.dbg, self.dbg),
                            'normalize': (self.normalize, self.inormalize),
                            'normalize_batch': (self.normalize_batch, self.inormalize_batch),
                            'resample_to_44100': (self.resample_to_44100, self.iresample_to_44100),
                            'spectrogram_phase': (self.spectrogram_96_phase, self.ispectrogram_96_phase),
                            'cast_int16_torch': (self.torch_cast_int16, self.dummy),
                            'sqrt(8/3)':(self.sqrt_8_3,self.isqrt_8_3)
                            }


        self.flag_direction = None # used for torubleshooting
        #self.kwargs = kwargs

        self.fs = 16000
        self.stft_n_fft = 32
                        


        # stores values from forward pass that are needed during the inverse pass
        self.phi = None     # stores angle of stft
        self.x_shape = None # stores size of x


        self.transforms_forward = list()
        self.transforms_inverse = list()
        
        for transform in transforms:
            self.transforms_forward.append(valid_transforms[transform][0]) # add all forward transforms to forward list    
            self.transforms_inverse.append(valid_transforms[transform][1]) # add their corresponding "inverse" to inverse list
            
        self.transforms_inverse.reverse() 

    def forward_torch(self,x):
        for t in self.transforms_forward:
            x = t(x)
        return x

    def forward(self, x):
        if torch.is_tensor(x):
            x = x.numpy()
        for t in self.transforms_forward:
            x = t(x)
        try:
            return torch.from_numpy(x).to(torch.float32)
        except:
            return x

    def inverse(self, x):
        if torch.is_tensor(x):
            x = x.numpy()
        
        for t in self.transforms_inverse:
            x = t(x)
        #x = (x*np.power(2,16)).astype(int)/np.power(2,16) # quantize to 16 bit
        
        return torch.from_numpy(x).to(torch.float32)

    def sqrt_8_3(self,x):
        return x * np.sqrt(8/3)

    def isqrt_8_3(self,x):
        return x / np.sqrt(8/3)

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


    def torch_cast_int16(self, x):
        return x.type(torch.int16)

            
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
        self.original_size_44100 = x.shape[1]
        self.nr_of_samples = int(x.shape[1] * (44100/self.fs))
        zero_pad_length = int((self.stft_n_fft/2)*np.ceil(self.nr_of_samples / (self.stft_n_fft/2)))
        
        x_44100 = np.zeros((x.shape[0],) + (zero_pad_length,))
        x_44100[:,:self.nr_of_samples] = signal.resample(x,self.nr_of_samples, axis=1)
        
        return x_44100

    def iresample_to_44100(self, x_44100):
        x = signal.resample(x_44100[:,:self.nr_of_samples], self.original_size_44100, axis=1)
        return x

    def mag2db(self, x):
        x[x == 0] = np.finfo(float).eps
        return 10 * np.log10(x)
    
    def db2mag(self, x):
        return np.power(10, x/10)

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
        f, t, s = signal.stft(x, nperseg=self.stft_n_fft, noverlap=self.stft_n_fft/2, nfft=self.stft_n_fft)
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
        if x.shape[0] == 1:
            return np.reshape(x, ((1,) + (1,) + (x.shape[1:])))

        else:
            x = np.squeeze(x)
            x_new = np.zeros((x.shape[0],) + (1,) + (x.shape[1:]),dtype=x.dtype)
            x_new[:,0] = x
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


    def spectrogram_96_phase(self,x):
        _,_,s = self.stft(x)
        self.phi = np.arctan2(s.real, s.imag)
        s_abs = np.abs(s)
        
        P = self.mag2db96(self.push_data_dim(s_abs))
        mag = self.db2mag(P)
        
        mag = self.pop_data_dim(mag)
        s_real = mag * np.sin(self.phi)
        s_imag = mag * np.cos(self.phi)
        s = s_real + s_imag*1.0j
        s = self.push_data_dim(s)
        
    
        return s

    def ispectrogram_96_phase(self, s):
        
        self.phi = np.arctan2(s.real, s.imag)
        mag = np.abs(s)

        P = self.mag2db(mag)
        s_abs = self.db2mag96(P)
        

        s_real = s_abs * np.sin(self.phi)
        s_imag = s_abs * np.cos(self.phi)        
        s = s_real + s_imag*1.0j
        
        s = self.pop_data_dim(s)
        x = self.istft(s) 
        return x

        

    def spectrogram_mel(self, x):
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
    
    def spectrogram32(self, x):
        f, t, s = signal.stft(x, nperseg=64, noverlap=48, padded=False)
        self.phi = np.arctan2(s.real, s.imag)
        s_abs = np.abs(s)
        return s_abs

    def ispectrogram32(self, s_abs):
        s_real = s_abs * np.sin(self.phi)
        s_imag = s_abs * np.cos(self.phi)
        s = s_real + s_imag*1.0j
        _, x = signal.istft(s, nperseg=64, noverlap=48)
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

        None

        
        
        

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


    