import numpy as np

from scipy import signal
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz

from graphics import get_audio
#_, audio = get_audio.get_audio_speech()
#audio = signal.resample(audio, int(audio.shape[0]*2.75))
#_, audio = get_audio.get_audio_sin()


NEGATIVE_INF = np.NINF

def get_mask_batch(x):
    mask = []
    for b in range(x.shape[0]): # iterate batches
        mask.append(get_masking_threshold(x[b]))

    return np.stack(mask)


def get_tonal_maskers(P_n):
    P_TM = np.zeros(P_n.shape) + NEGATIVE_INF
    P_TM_lm = signal.argrelextrema(P_n, np.greater, axis=0)[0] # get index of all local maximas
    
    for idx in P_TM_lm:
        masker = True
        if 1 < idx < 62: # check what bark range to use
            dk_l = [-2, 2]
        elif 61 < idx < 127:
            dk_l = [-3,-2, 2, 3]
        elif 126 < idx < 249:
            dk_l = [-6,-5,-4,-3,-2,2,3,4,5,6]
        else:
            masker = False
            dk_l = []
        
        for dk in dk_l:
            if P_n[idx] - P_n[idx+dk] <= 7:
                masker = False
                break

        if masker == True:
            P_TM[idx] = mag2db(db2mag(P_n[idx-1]) + db2mag(P_n[idx]) + db2mag(P_n[idx+1]))
        
    return P_TM

def get_nontonal_maskers(P_n, CB, exclude_idx, f_steps):
    
    NR_BINS = len(CB) - 1
    NM = np.zeros((NR_BINS))
    
    for n in range(NR_BINS):
        low = CB[n] 
        high = CB[n + 1]
        
        idxs = np.logical_and(np.array([f_steps - low >= 0]), np.array([f_steps - high < 0]))[0]
        idxs[exclude_idx] = False
    
        NM[n] = np.sum(db2mag(P_n[idxs]))
    return mag2db(NM)

def quiet_threshold(f):
    # output is in dB
    if np.isscalar(f):
        if f == 0:
            f = np.finfo(float).eps
    else:
        f[f==0] = np.finfo(float).eps

    threshold = 3.64*np.power(f/1000, -0.8) \
        - 6.5 * np.exp(-0.6*(np.square((f/1000) - 3.3))) \
        + 1e-3 * np.power(f/1000, 4)
    threshold[threshold>96] = 96
    return threshold

def TM_offset_db(masker):
    return -6.025 - 0.275*masker

def NM_offset_db(masker):
    return -2.025 - 0.175*masker


def spreading_function_db(masker, maskee, val):
    dz = masker - maskee
    if -3 < dz < -1:
        SF_db = 17*dz - 0.4 * val + 11
    elif -1 < dz < 0:
        SF_db = (0.4*val +6) * dz
    elif 0 < dz < 1:
        SF_db = -17*dz
    elif 1 < dz < 8:
        SF_db = -17*dz+0.15 *val*(dz-1)
    else:
        SF_db = 0
    return SF_db


def mag2db(x):
    if np.isscalar(x):
        if x == 0:
            return np.finfo(float).eps
        return 10 * np.log10(x)
    else:
        x[x == 0] = np.finfo(float).eps
        return 10 * np.log10(x)

def db2mag(x):
    return np.power(10, x/10)


def f_to_bark(f):
    return 13*np.arctan(0.76*f / 1000) + 3.5*np.square(np.arctan(f/7500))

def get_masking_threshold(x):   
    Fs = 44100
    CB_f = [0,100,200,300,400,510,630,770,920, 1080,1270,1480,1720,2000,2320,2700,3150, 3700,4400,5300,6400,7700,9500,12000,15500]
    CB_f_mean = np.array([(CB_f[i] + CB_f[i+1])/2 for i in range(len(CB_f)-1)])
    
    # constants
    N = 512

    # ============================= STEP 1 =============================

    # modified hann window, eq 2.9
    n = np.arange(N)
    w = np.sqrt(8/3) * (1/2) * (1 - np.cos(2*np.pi*n/N))

    

    f_steps, t_steps, S = signal.stft(x, nperseg=N, fs=Fs, noverlap=384)
    bark_steps = f_to_bark(f_steps)
    max_bark = min(np.max(bark_steps), 25)

    S = np.square(np.abs(S))
    S[S==0] = np.finfo(float).eps
    PSD = 10 * np.log10(S)
    P = 96 - np.max(PSD) + PSD

    
    #MASK_256 = np.zeros(P.shape[0])
    
    MASK_32 = np.zeros((32, P.shape[1]))
    #MASK_db = np.zeros(P.shape) - NEGATIVE_INF

    # ============================= STEP 2 =============================
    # perform same analysis on every time step
    for t_idx in range(t_steps.shape[0]):
        
        P_n = P[:,t_idx]

        P_TM = get_tonal_maskers(P_n) # 257 bins with only tonal maskers Power remaining
        exclude_idx = np.where(P_TM > NEGATIVE_INF)[0]
        
        P_NM = get_nontonal_maskers(P_n, CB_f, exclude_idx,f_steps) # 24 bins of nontonal maskers
        
        # ========================= STEP 3 ===============================
        # remove maskers weaker then the hearing threshold in quiet
        P_quiet_TM = quiet_threshold(f_steps)
        P_TM[P_TM < P_quiet_TM] = NEGATIVE_INF
        
        P_quiet_NM = quiet_threshold(CB_f_mean)
        P_NM[P_NM < P_quiet_NM] = NEGATIVE_INF


        # remove weak maskers within 0.5 bark
        
        TM = P_TM>NEGATIVE_INF
        order = np.argsort(P_TM)[::-1]
        bark_of_max_TM = f_to_bark(f_steps[order][TM])
        bark_of_NM = f_to_bark(CB_f_mean)
        for k, bark in enumerate(bark_of_max_TM):
            # check closest TM
            
            TM_val = P_TM[TM][k]
            TM_val_within_range = np.where(np.abs(bark - bark_of_max_TM) < 0.5)[0] # all of these lay within 0.5 bark
            
            # TODO start with strongest
            # np.sort(TM_val[TM_val_within_range]) # sort so we check strongest value first
            
            
            # compare TM_val to all within 0.5 bark range
            for idx in TM_val_within_range:
                if TM_val < P_TM[order][idx]:
                    P_TM[order][k] = NEGATIVE_INF
            


            # also check closest NM (we only look at closest since we know these are spread ~1 bark apart)
            NM_idx = np.argmin(np.abs(bark_of_NM - bark))
            NM_val = P_NM[NM_idx]
            if TM_val < NM_val:
                P_TM[TM][idx] = NEGATIVE_INF
            else:
                P_NM[NM_idx] = NEGATIVE_INF
            
        
        # ========================= STEP 4 ===============================
        
    
        MASK_106 = np.zeros((106))
        
        maskees = list(range(0,48,1))
        maskees.extend(list(range(48,96,2)))
        maskees.extend(list(range(96,232,4)))
        maskees_hz = np.array(maskees)*Fs/2/256

        maskees_bark = f_to_bark(maskees_hz)


        for k, maskee_bark in enumerate(maskees_bark):
            # get TM for k
            for idx in np.where(P_TM>NEGATIVE_INF)[0]:
                masker_bark = f_to_bark(idx * Fs/2 * 256)
                if (masker_bark - maskee_bark < 8) and (masker_bark - maskee_bark > -3):               
                    MASK_106[k] += db2mag(P_TM[idx]
                                + TM_offset_db(masker_bark)
                                + spreading_function_db(masker_bark, maskee_bark, P_TM[idx]))
            
            # get NM for k
            for idx in np.where(P_NM>NEGATIVE_INF)[0]:
                masker_bark = bark_of_NM[idx]
                if (masker_bark - maskee_bark < 8) and (masker_bark - maskee_bark > -3):
                    contribution = db2mag(P_NM[idx]
                                + NM_offset_db(masker_bark)
                                + spreading_function_db(masker_bark, maskee_bark, P_NM[idx]))

                    MASK_106[k] += contribution
                    

        # get threshold in quiet for k
        MASK_106 += db2mag(quiet_threshold(maskees_hz))
            
        # ========================= STEP 5 ===============================
        MASK_106 = mag2db(MASK_106)
        #MASK_106[0] = 0
        # ========================= STEP 6 ===============================
        # pick out min from each subband

        MASK_32[0:6,t_idx] = np.min(np.reshape(MASK_106[0:48],(6,8)), axis=1)
        MASK_32[6:12,t_idx] = np.min(np.reshape(MASK_106[48:72],(6,4)), axis=1)
        MASK_32[12:29,t_idx] = np.min(np.reshape(MASK_106[72:106],(17,2)), axis=1)
        MASK_32[29:,t_idx] = MASK_106[105]
        
        #plt.plot(f_steps,P_n)
        #plt.plot(Fs/2/32*np.linspace(0,32,32),MASK_32[:,t_idx])
        #plt.plot(f_steps[1:],quiet_threshold(f_steps[1:]))
        #plt.show()
        MASK_32[MASK_32 > 96] = 96
    return MASK_32


#get_masking_threshold(audio[8000:])
#get_masking_threshold(audio[8000:])