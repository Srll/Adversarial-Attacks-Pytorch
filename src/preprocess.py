import librosa
import numpy as np
from scipy import signal
import torch
import matplotlib.pyplot as plt 
# TODO add MFCC, MelSPECTROGRAM, delta calculations for MFCC


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
                            'normalize': (self.normalize, self.inormalize)}


        self.flag_direction = None # used for torubleshooting
        #self.kwargs = kwargs
    
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
        self.flag_direction = 'forward'
        x = x.numpy()
        for t in self.transforms_forward:
            x = t(x)
        self.flag_direction = 'done_forward'
        return torch.from_numpy(x).to(torch.float32)

    def inverse(self, x):
        x = x.numpy()
        self.flag_direction = 'inverse'
        for t in self.transforms_inverse:
            x = t(x)
        self.flag_direction = 'done_inverse'
        return torch.from_numpy(x).to(torch.float32)

    """def normalize(self, x):
    """

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
        self.x_shape = x.shape
        s = list()
        for b in range(x.shape[0]):
            s.append(librosa.stft(x[b,:], n_fft=self.kwargs.get('stft_n_fft')))
        s = np.stack(s)
        
        return s

    def istft(self, s):
        x = np.zeros(self.x_shape)
        #s_ = s[..., 0] + s[..., 1]*1.0j # add imag and real dimensions together
        #input(s_.shape)
        for b in range(s.shape[0]):
            x[b] = librosa.istft(s[b,:,:])
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
        return x_new

    def spectrogram(self, x):
        
        """pad = wargs.get('spectrogram_pad')
        if pad > 0:
            x = torch.nn.functional.pad(x, (pad, pad), "constant")
        """
        s = self.stft(x)
        self.phi = np.arctan2(s.real,s.imag)
        s_pow = np.sqrt(np.power(s.real,2) + np.power(s.imag,2))
        
        return s_pow

    def ispectrogram(self, s_pow):
        
        s_real = s_pow * np.cos(self.phi)
        s_imag = s_pow * np.sin(self.phi)
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