import numpy as np
from get_audio import get_audio_sin, get_audio_speech
from scipy import signal
from matplotlib import pyplot as plt

def mag2db(x):
    x[x == 0] = np.finfo(float).eps
    return 10 * np.log10(x)

def db2mag(x):
    return np.power(10, x/10)

def quiet_threshold(f):
    # output is in dB
    threshold = 3.64*np.power(f/1000, -0.8) \
        - 6.5 * np.exp(-0.6*(np.square((f/1000) - 3.3))) \
        + 1e-3 * np.power(f/1000, 4)
    return threshold

def local_maximas(p):
    # returns index of all local extrema over frequency axis
    return np.stack(signal.argrelextrema(p, np.greater, axis=0), axis=-1) 

def f_to_bark(f):
    return 13*np.arctan(0.76*f / 1000) + 3.5*np.square(np.arctan(f/7500))

def bark_to_f(bark):
    return 

# critical bands over frequency spectrum
CB_f = [0,100,200,300,400,510,630,770,920, \
        1080,1270,1480,1720,2000,2320,2700,3150, \
        3700,4400,5300,6400,7700,9500,12000,15500]


fs,x = get_audio_sin()

# constants
N = 512

# ============================= STEP 1 =============================

# modified hann window, eq 2.9
n = np.arange(N)
w = np.sqrt(8/3) * (1/2) * (1 - np.cos(2*np.pi*n/N))

f_steps, t_steps, S = signal.stft(x, nperseg=N, fs=fs)
bark_steps = f_to_bark(f_steps)
PSD = 10 * np.log10(np.square(np.abs(S)))
P = 96 - np.max(PSD) + PSD # should this be over one FFT or over STFT?

# ============================= STEP 2 =============================
for t_idx in range(t_steps.shape[0]):
    P_lm = local_maximas(P)
    
    S_TM = []
    for i in range(P_lm.shape[0]):
        masker = True
        if 1 < P_lm[i][0] < 62: # check what bark range to use
            dk_l = [-2, 2]
        elif 61 < P_lm[i][0] < 127:
            dk_l = [-3,-2, 2, 3]
        elif 126 < P_lm[i][0] < 249:
            dk_l = [-6,-5,-4,-3,-2, 2,3,4,5,6]
        else:
            masker = False
            dk_l = []
        
        for dk in dk_l:


            if P[P_lm[i][0],P_lm[i][1]] - P[P_lm[i][0]+dk,P_lm[i][1]] <= 7:
                masker = False

                break
            
        if masker == True:
            S_TM.append([P_lm[i][0],P_lm[i][1]])


    S_TM = np.asarray(S_TM)

    P_TM = mag2db(db2mag(P[S_TM[:,0]-1, S_TM[:,1]]) + db2mag(P[S_TM[:,0], S_TM[:,1]]) + db2mag(P[S_TM[:,0]-1, S_TM[:,1]]))


    #P_TM = 10*np.log10(np.power(10,/10) \
    #                    + np.power(10,/10) \
    #                    + np.power(10,/10))
    
    """
    plt.plot(S_TM[S_TM[:,1]==4][:,0], P_TM[S_TM[:,1]==4],'ro')
    plt.plot(P[:,4])
    plt.show()
    """

    bins_e = np.zeros((len(CB_f), S.shape[1]))
    cb_l = 0
    
    for i, cb_h in enumerate(CB_f):
        bin_sum  = np.zeros(S.shape[1])
        for f_idx in range(f_steps.shape[0]):
            f = f_steps[f_idx]
            if (cb_h<f):
                bins_e[i] = mag2db(bin_sum)
                break
            elif(cb_l<f):
                bin_sum += db2mag(P[f_idx,:])
        cb_l = cb_h

    
    bins_f = np.zeros(len(CB_f) + 1)
    
    bark_list = (bark_steps//1).astype(int).tolist()
    for i,idx in enumerate(bark_list):
        bins_f[idx] += np.log10(f_steps[i])
    bins_f /= np.unique(bark_list, return_counts=True)[1]
    bins_f = np.power(10, bins_f)

    #bins_f = np.power(10, bins_f)
    #bins_f = db2mag(bins_f)

    """
    plt.plot(f_steps[S_TM[S_TM[:,1]==4][...,0]], P_TM[S_TM[:,1]==4],'go')
    plt.plot(bins_f[:-1], bins_e[:,4],'ro')
    plt.plot(f_steps,P[:,4])
    plt.show()
    """
    


    

    # ============================= STEP 3 =============================
    
    # check if SPL is higher then quiet threshold
    P_NM_idx = (quiet_threshold(bins_f[:-1]) < bins_e.T).T  # noise
    P_NM = np.zeros((25,S.shape[1]))  # TODO add constant -inf ?
    P_NM[P_NM_idx == True] = bins_e[P_NM_idx]

    #import pdb; pdb.set_trace()
    P_TM_idx = quiet_threshold(f_steps[S_TM[...,0]]) < P_TM   # tonal
    S_TM = S_TM[P_TM_idx]
    #print(S_TM.shape)
    
    pop_idx = []
    for i in range(S.shape[1]):
        # bark of max values at time = i 
        bark_of_max = f_to_bark(f_steps[S_TM[i == S_TM[...,1]][...,0]])
        bark_diff = np.diff(bark_of_max)
        for k, diff in enumerate(bark_diff):
            if diff <= 0.5:
                # check which max to keep
                value1 = P_TM[S_TM[i == S_TM[...,1]][...,0]][k]
                value2 = P_TM[S_TM[i == S_TM[...,1]][...,0]][k+1]
                if value1 > value2:
                    # add idx to pop list 
                    pop_idx.append([S_TM[i == S_TM[...,1]][...,0][k],i])
                else:
                    pop_idx.append([S_TM[i == S_TM[...,1]][...,0][k+1],i])
            
    for idx in pop_idx:
        for i in np.where(S_TM[...,0] == idx[0])[0].tolist():
            if S_TM[i][1] == idx[1]:
                pop = i
        
        S_TM = np.delete(S_TM, pop, 0)
        P_TM = np.delete(P_TM, pop, 0)
    """
    plt.plot(f_steps[S_TM[S_TM[:,1]==4][...,0]], P_TM[S_TM[:,1]==4],'go')
    plt.plot(bins_f[:-1], bins_e[:,4],'ro')
    plt.plot(f_steps,P[:,4])
    plt.show()
    """


    #f_to_bark(f_steps[S_TM[P_TM_idx][...,0]])


    # ============================= STEP 4 =============================
    nr_maskees = 257
    maskees = 25*np.arange(1,nr_maskees+1)/nr_maskees
    mask_frame_TM = np.zeros((nr_maskees, S.shape[1]))

    mask_frame_NM = np.zeros((nr_maskees, S.shape[1]))
    
    for i in range(P_TM.shape[0]):
        value  = P_TM[i]
        idxs = S_TM[i]
        idx_bark   = f_to_bark(f_steps[idxs[0]])
        for j in range(maskees.shape[0]):
            dz = maskees[j] - idx_bark
            if -3 < dz < -1:
                SF_db = 17*dz - 0.4 * value + 11
            elif -1 < dz < 0:
                SF_db = (0.4*value +6) * dz
            elif 0 < dz < 1:
                SF_db = -17*dz
            elif 1 < dz < 8:
                SF_db = -17*dz+0.15 *value*(dz-1)
            else:
                SF_db = -10000 # TODO
            
            mask_frame_TM[j, idxs[1]] += db2mag(SF_db -6.025 - 0.275*maskees[j] + value)
    mask_frame_TM = mag2db(mask_frame_TM)
    
    for i in range(P_NM.shape[0]):
        values  = P_NM[i]
        source_bark = i
        for k in range(values.shape[0]):
            value = values[k]
            for j in range(maskees.shape[0]):
                dz = maskees[j] - source_bark 
                if -3 < dz < -1:
                    SF_db = 17*dz - 0.4 * value + 11
                elif -1 < dz < 0:
                    SF_db = (0.4*value +6) * dz
                elif 0 < dz < 1:
                    SF_db = -17*dz
                elif 1 < dz < 8:
                    SF_db = -17*dz+0.15 *value*(dz-1)
                else:
                    SF_db = -10000 # TODO
                # TODO not sure about this summation, is everything in dB?
                #mask_frame_NM[i, k] += np.power(10, (np.power(10,SF_db/10) -6.025 - 0.275*maskees[j] + value)/10)
                mask_frame_NM[j, k] += db2mag(SF_db -2.025 - 0.175*maskees[j] + value)
    mask_frame_NM = mag2db(mask_frame_NM)
    
    L_G = mag2db(db2mag(mask_frame_TM) + db2mag(mask_frame_NM))
    
    plt.plot(maskees,L_G[:,10])
    plt.plot(maskees,mask_frame_NM[:,10])
    plt.plot(maskees,mask_frame_TM[:,10])
    plt.plot(maskees,P[:,10])
    plt.show()
    

    # ============================= STEP 5 =============================



        







