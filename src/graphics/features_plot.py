import numpy as np
import librosa as lb
from scipy.io import wavfile
import matplotlib.pyplot as plt



def get_audio_speech():
    fs, x = wavfile.read("00f0204f_nohash_0.wav")
    x = x / np.max(x)
    return fs, x

fs, x = get_audio_speech()
y_res = 3000

x_mat = np.zeros((y_res, x.shape[0]))
for i in range(x.shape[0]):
    temp = int(y_res/2 + int(x[i] * (y_res/2.1)))
    x_mat[temp,i] = 1

melspectrogram = lb.power_to_db(lb.feature.melspectrogram(y=x, sr=fs, n_mels=256,fmax=8000))
melspectrogram_up = (melspectrogram.repeat(int(x.shape[0]/melspectrogram.shape[1]),axis=1)).repeat(int(y_res/melspectrogram.shape[0]), axis=0)

mfcc = lb.feature.mfcc(y=x, sr=fs, n_mels=20,fmax=8000)
mfcc_up = (mfcc.repeat(int(x.shape[0]/mfcc.shape[1]),axis=1)).repeat(int(y_res/mfcc.shape[0]),axis=0)


fig, axes = plt.subplots(nrows=3, ncols=1,sharex='all',sharey='all')

axes[0].matshow(x_mat.T)
axes[0].set_title('Time Domain')
axes[1].matshow(melspectrogram)
axes[1].set_title('Melspectrogram')
axes[2].matshow(mfcc)
axes[2].set_title('MFCC')
plt.show()
"""plt.subplot(311)
#plot.title('Spectrogram of a wav file with piano music')
plt.plot(x)
plt.xlabel('Sample')
plt.ylabel('Amplitude')


plt.subplot(312)
plt.matshow(mfcc)
#plt.imshow(mfcc_up)
plt.subplot(313)
plt.matshow(melspectrogram)
#plt.imshow(melspectrogram_up)
plt.show()
"""