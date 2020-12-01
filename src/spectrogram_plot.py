from scipy.io import wavfile
import numpy as np
import preprocess
from matplotlib import pyplot as plt

def get_spectrogram():
    nem = "17"
    fs, x = get_audio_clean(nem)
    fs, x_adv = get_audio_adv(nem)
    x_noise = x - x_adv
    
    
    
    T = preprocess.PreProcess(['spectrogram','insert_data_dim','mag2db'])
    T.stft_n_fft = 128
    
    
    z = T.forward(np.expand_dims(x, axis=0))
    z_adv = T.forward(np.expand_dims(x_adv,axis=0))
    z_noise = T.forward(x_noise)

    min_clip = -40
    z = np.clip(z,min_clip,10000)
    z_adv = np.clip(z_adv,min_clip,10000)
    z_noise = np.clip(z_noise,min_clip,10000)


    minima = np.array([z.min(),z_adv.min(),z_noise.min()]).min()
    

    
    z_squ = np.squeeze(z) - minima
    z_adv_squ = np.squeeze(z_adv) - minima
    z_noise_squ = np.squeeze(z_noise) - minima

    maxima = np.array([z_squ.max(),z_adv_squ.max(),z_noise_squ.max()]).max()

    #fig=plt.figure(figsize=(8, 8))
    fig=plt.figure()
    rows = 3
    import pdb; pdb.set_trace()
    
    # clean
    fig.add_subplot(rows, 1, 1)
    #plt.imshow(z_squ/maxima, origin="lower",vmin=0, vmax=1)
    plt.specgram(x, Fs=fs)
    # adv
    fig.add_subplot(rows, 1, 2)
    #plt.imshow(z_adv_squ/maxima, origin="lower",vmin=0, vmax=1)
    plt.specgram(x_adv, Fs=fs)
    # noise
    fig.add_subplot(rows, 1, 3)
    #plt.imshow((z_squ/maxima - z_adv_squ/maxima) * 1/(z_squ/maxima - z_adv_squ/maxima).max(), origin="lower",vmin=0, vmax=1)
    #plt.imshow(z_noise_squ/maxima, origin="lower",vmin=0, vmax=1)
    plt.specgram(x_noise, Fs=fs)
    plt.show()

    
    fig_z=plt.figure()
    plt.imshow(z, origin="lower",vmin=0, vmax=maxima)
    plt.savefig("spectrogram_x.pdf")
    fig_z_adv=plt.figure()
    plt.imshow(z_adv, origin="lower",vmin=0, vmax=maxima)
    plt.savefig("spectrogram_x_adv.pdf")
    fig_z_noise=plt.figure()
    plt.imshow(z_noise, origin="lower",vmin=0, vmax=maxima)
    plt.savefig("spectrogram_x_noise.pdf")


def get_audio_clean(n):
    fs, x = wavfile.read("x"+str(n)+".wav")
    x = x / np.max(x)
    return fs, x

def get_audio_adv(n):
    fs, x = wavfile.read("x"+str(n)+"_adv.wav")
    x = x / np.max(x)
    return fs, x

    

def get_audio_speech():
    fs, x = wavfile.read("00f0204f_nohash_0.wav")
    x = x / np.max(x)
    return fs, x

get_spectrogram()