from inaudible import preprocess, masking
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import pickle
import os.path

def generate_adversarial_RGAP(self,x, y, targeted=False, eps=1, verbose=False,adv_parameters=[20001]):
    # adv_parameters = [N_iterations]
    import pdb; pdb.set_trace()
    N_iterations = int(adv_parameters[0])
    def mag2db(xi):
        if np.isscalar(xi):
            if xi == 0:
                return np.finfo(float).eps
            return 10 * np.log10(xi)
        else:
            xi[xi == 0] = np.finfo(float).eps
            xi = 10 * np.log10(xi)
            xi[xi>96] = 96
            return xi

    def db2mag(xi):
        return np.power(10, xi/10)

    
    
    PLOT = True
    EARLY_STOP = True
    FS_MODEL = 16000
    FS_Z = 44100
    

    F_RESOLUTION = int(self.adversarial_preprocess.stft_n_fft/2) + 1
    F_MIN = 0 #int(np.floor(500 / (FS_Z/2) * F_RESOLUTION))              # limit lowest frequency perturbation to 500 hz
    F_MAX = int(np.ceil(F_RESOLUTION * FS_MODEL/(FS_Z)))    # limit highest frequency perturbation to half of models sampling frequency
    
    self.model.eval()
    x_original = x.clone().detach() # save for evaluation at the end
    z_invertable = self.adversarial_preprocess.forward(x) # save inverse transform for later 
    
    z_2d = signal.resample(z_invertable,F_RESOLUTION,axis=2)
    N_BATCH = z_2d.shape[0]
    MAX_POS = z_2d.shape[2:]

    accuracy = np.zeros((N_BATCH, 0))
    

    z_1d_original = np.reshape(z_2d, (N_BATCH,1,np.prod(MAX_POS)))
    z_1d_adv = np.copy(z_1d_original)
    
    print("Calculating masking threshold")
    m_32 = masking.get_mask_batch(signal.resample(x_original.numpy(), int(x.shape[1] * (FS_Z/FS_MODEL)), axis=1))

    COLOR_LIST = ['salmon','olive','darkgreen','khaki','black','grey','orange','maroon','sandybrown','lightblue','purple','pink','yellow','royalblue','tan','cyan','blue','red','violet','silver','gold']    
    
    # MMT processing
    m_2d = signal.resample(m_32,MAX_POS[1],axis=-1)     # resample to same time resolution
    m_2d = signal.resample(m_2d,F_RESOLUTION,axis=1)    # resample to same frequency resolution
    m_1d = np.reshape(m_2d, (N_BATCH, np.prod(MAX_POS)))
    m_1d[m_1d > 96] = 96
    m_1d_mag = db2mag(m_1d)

    with torch.no_grad():
        x.detach()
        pred_best = np.array(np.diag(torch.nn.functional.softmax(self.model(x), dim=1).numpy()[:,y]))

    if verbose:
        plt.axis([0, N_iterations, 0, 1])
        plt.ion()
    
    available_idx = np.tile(np.arange(F_MIN, F_MAX) * F_RESOLUTION, (MAX_POS[1],1)).T + np.arange(MAX_POS[1])
    unperturbed_idx_list = np.reshape(available_idx, MAX_POS[1]*(F_MAX-F_MIN)).astype(dtype=np.int16)
    
    perturb_idx = np.array([np.random.choice(unperturbed_idx_list, N_iterations, replace=True) for i in range(N_BATCH)])
    active = np.arange(0,N_BATCH)
    
    if os.path.isfile(os.path.join('save','RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_n'+'.pickle')):
        n_start = pickle.load(open(os.path.join('save','RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_n'+'.pickle'),'rb') )
    else:
        n_start = 0
    if os.path.isfile(os.path.join('save','RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_z_1d_adv'+'.pickle')):
        z_1d_adv = pickle.load(open(os.path.join('save','RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_z_1d_adv'+'.pickle'),'rb') )

    for n in range(n_start, N_iterations):
        if (n % 1000) == 0: # save state
            pickle.dump(n,open(os.path.join('save','RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_n'+'.pickle'),'wb'))
            pickle.dump(z_1d_adv,open(os.path.join('save','RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_z_1d_adv'+'.pickle'),'wb'))
                    
        z_1d_try = z_1d_adv.copy()
        mult = np.random.uniform(-eps,eps, size=(len(active)))

        for i,b in enumerate(list(active)):
            z_1d_try[b,0,perturb_idx[b, n]] = mag2db(np.abs(db2mag(z_1d_original[b,0,perturb_idx[b, n]]) + mult[i]*m_1d_mag[b,perturb_idx[b,n]]))
                    
        z_2d_active = np.reshape(z_1d_try[active], (len(active), 1, MAX_POS[0], MAX_POS[1]))
        z_2d_invertable = signal.resample(z_2d_active, z_invertable.shape[2], axis=2)

        x_try = self.adversarial_preprocess.inverse(z_2d_invertable)


        with torch.no_grad():
            pred_try = np.array(np.diag(torch.nn.functional.softmax(self.model(x_try), dim=1).numpy()[:,y[active]]))
        if not targeted:
            idxs = np.nonzero(pred_try < pred_best[active])[0]
        else:
            idxs = np.nonzero(pred_try > pred_best[active])[0]

        z_1d_adv[active[idxs]] = z_1d_try[active[idxs]]
        pred_best[active[idxs]] = np.squeeze(pred_try[idxs])
        
        if EARLY_STOP:
            if not targeted:
                active_temp = np.nonzero(pred_best > 0.1)[0]
                if len(active_temp) == 0:
                    break
                if len(active) != len(active_temp): # if change in active dims, update variables
                    active = active_temp
                    self.adversarial_preprocess.forward(x[active]) # update local variables in preprocessing                        
            else:
                active_temp = np.nonzero(pred_best < 0.9)[0]
                if len(active_temp) == 0:
                    break
                if len(active) != len(active_temp): # if change in active dims, update variables
                    active = active_temp
                    self.adversarial_preprocess.forward(x[active]) # update local variables in preprocessing
        
        if verbose:
            if n % 1000 == 0:
                
                accuracy = np.hstack((accuracy, np.expand_dims(pred_best, -1)))
                
            if n % int(N_iterations/20) == 0:
                print("-----------------")
                print(pred_best[active])
                print(pred_try)
                if PLOT:
                    for b in list(active):
                        plt.scatter(n, pred_best[b],c=COLOR_LIST[int(20*b/N_BATCH)])
                plt.show()
                plt.pause(0.0001)
            plt.savefig('output\\Accuracy_RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted) + '.png')
        
    with open('output\\U_count_RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted) + '.txt','ab+') as f:
        np.savetxt(f, np.count_nonzero(z_1d_original - z_1d_adv,axis=(1,2)))
    with open('output\\Accuracy_RG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted) + '.csv','ab+') as f:
        np.savetxt(f, accuracy, delimiter=",")
        
    z_2d = np.reshape(z_1d_adv, (N_BATCH, 1, MAX_POS[0], MAX_POS[1]))
    z_257 = signal.resample(z_2d, z_invertable.shape[2], axis=2)
    self.adversarial_preprocess.forward(x_original) # recalculate phase information
    x_adv = self.adversarial_preprocess.inverse(z_257).detach()
    

    with torch.no_grad(): 
        y_estimate_adversarial = torch.nn.functional.softmax(self.model(x_adv),dim=1)
        y_estimate = torch.nn.functional.softmax(self.model(x_original.to(torch.float32)),dim=1)
    noise = x_adv - x_original
    return x_adv.to(torch.float32), noise.to(torch.float32), y_estimate_adversarial.to(torch.float32), y_estimate.to(torch.float32)
