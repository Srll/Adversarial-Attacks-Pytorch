from inaudible import preprocess, masking
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import pickle
import os.path
import progressbar

def generate_adversarial_LGAP(self,x, y, targeted=False, eps=1, verbose=False, adv_parameters=[20,1000]):
    # adv_parameters = [N_iterations, N_perturbations]

    N_loops = int(adv_parameters[0])
    N_pixels = int(adv_parameters[1])
    
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
    
    FS_MODEL = 16000
    FS_Z = 44100
    F_RESOLUTION = int(self.adversarial_preprocess.stft_n_fft/2) + 1
    F_MAX = int(np.ceil(F_RESOLUTION * FS_MODEL/(FS_Z)))    # limit highest frequency perturbation to half of models sampling frequncy
    
    self.model.eval()
    x_original = x.clone().detach() # save for evaluation at the end
    z = self.adversarial_preprocess.forward(x)

    COLOR_LIST = ['salmon','olive','darkgreen','khaki','black','grey','orange','maroon','sandybrown','lightblue','purple','pink','yellow','royalblue','tan','cyan','blue','red','violet','silver','gold']

    z_shape = z.shape
    T_RESOLUTION = z.shape[-1]
    N_BATCH = z_shape[0]
    accuracy = np.zeros((N_BATCH, 0))
    
    with torch.no_grad():
        pred_best = np.array(np.diag(torch.nn.functional.softmax(self.model(x_original.numpy()), dim=1).numpy()[:,y]))
    accuracy = np.hstack((accuracy, np.expand_dims(pred_best, -1)))

    z_2d = signal.resample(z,F_RESOLUTION,axis=2)
    


    m = masking.get_mask_batch(signal.resample(x_original.numpy(), int(x.shape[1] * (FS_Z/FS_MODEL)), axis=1))
    
    m_2d = signal.resample(m, F_RESOLUTION, axis=1)
    m_2d = signal.resample(m_2d, T_RESOLUTION, axis=2)
    m_2d_mag = db2mag(m_2d)
    
    N_dimensions_position = len(z_shape) - 2 # subtract batch and data dim
    max_pos = z_2d.shape[2:]
    
    EARLY_STOP = True
    active = np.arange(0,N_BATCH)
    N_neighbors = 5
    PLOT = True

    if verbose:
        plt.axis([0, N_loops, 0, 1])
        plt.ion()
    
    def add_pixels(z, pixels, active):
        z_perturbed = z.copy()
        z_perturbed[pixels[:,:,0],pixels[:,:,1],pixels[:,:,2],pixels[:,:,3]] = mag2db(np.abs(db2mag(z_perturbed[pixels[:,:,0],pixels[:,:,1],pixels[:,:,2],pixels[:,:,3]]) + pixels[:,:,4] * m_2d_mag[pixels[:,:,0],pixels[:,:,2],pixels[:,:,3]]))
        return z_perturbed[active]

    # initialize pixels
    pixels = np.random.rand(N_BATCH, N_pixels, 5) # each pixel has two dimensions (x_pos, y_pos)
    
    if os.path.isfile(os.path.join('save','LG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_pixels'+'.pickle')):
        pixels = pickle.load(open(os.path.join('save','LG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_pixels'+'.pickle'), "rb" ))
    else:
        pixels[:,:,0] = np.repeat(np.expand_dims(np.arange(0,N_BATCH),axis=1),N_pixels,axis=1)
        pixels[:,:,1] = 0
        pixels[:,:,2] = pixels[:,:,2] * (F_MAX - 2)
        pixels[:,:,3] = pixels[:,:,3] * (max_pos[1] - 2)
        pixels[:,:,4] = pixels[:,:,4] * 2 * eps - eps
        pixels = pixels.astype(np.int)
    pixels_neighbors = np.zeros(pixels.shape[:-1] + (N_neighbors,) + (5,), dtype=np.int)
    if os.path.isfile(os.path.join('save','LG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_n'+'.pickle')):
        n_start = pickle.load(open(os.path.join('save','LG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_n'+'.pickle'), "rb" ) )
    else:
        n_start = 0
    

    with torch.no_grad():
        z_adv = signal.resample(add_pixels(z_2d,pixels,active),z_shape[2],axis=2)
        x_pixels = self.adversarial_preprocess.inverse(z_adv).detach()
        pred_best = np.array(np.diag(torch.nn.functional.softmax(self.model(x_pixels), dim=1).numpy()[:,y[active]]))



    # main optimization loops
    for i in progressbar.progressbar(range(n_start, N_loops), redirect_stdout=True):
        # pixels neighbor f, t, neighbor_number, (batch, 0, x, y, specific_eps)
        pickle.dump(i, open(os.path.join('save','LG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_n'+'.pickle'),"wb"))
        pickle.dump(pixels, open(os.path.join('save','LG_N'+str(F_RESOLUTION)+'_'+str(int(eps))+str(targeted)+'_pixels'+'.pickle'),"wb"))
        
        pixels_neighbors[:,:,:,0] = np.repeat(np.expand_dims(pixels[:,:,0], axis=-1), 5, axis=2)
        pixels_neighbors[:,:,:,1] = 0

        pixels_neighbors[:,:,0,2] = pixels[:,:,2] + 1
        pixels_neighbors[:,:,0,3] = pixels[:,:,3]

        pixels_neighbors[:,:,1,2] = pixels[:,:,2] - 1
        pixels_neighbors[:,:,1,3] = pixels[:,:,3]

        pixels_neighbors[:,:,2,2] = pixels[:,:,2]
        pixels_neighbors[:,:,2,3] = pixels[:,:,3] + 1

        pixels_neighbors[:,:,3,2] = pixels[:,:,2]
        pixels_neighbors[:,:,3,3] = pixels[:,:,3] - 1
        
        pixels_neighbors[:,:,4,2] = np.random.randint(0, F_MAX-1, size=pixels_neighbors[:,:,4,2].shape)
        pixels_neighbors[:,:,4,3] = np.random.randint(0, max_pos[1]-1, size=pixels_neighbors[:,:,4,3].shape)
        
        pixels_neighbors[pixels_neighbors[:,:,:,2] > (F_MAX - 1)] -= np.array([0,0,1,0,0])
        pixels_neighbors[pixels_neighbors[:,:,:,3] > (max_pos[1] - 1)] -= np.array([0,0,0,1,0])

        pixels_neighbors[:,:,:,4] = np.random.uniform(-eps,eps, size=((N_BATCH,N_pixels,N_neighbors)))

        # optimize each pixel at a time
        for p in range(N_pixels):
            # calculate prediction with current pixel positions
            with torch.no_grad():
                z_adv = signal.resample(add_pixels(z_2d,pixels,active),z_shape[2],axis=2)
                x_pixels = self.adversarial_preprocess.inverse(z_adv).detach()
                pred_best[active] = np.array(np.diag(torch.nn.functional.softmax(self.model(x_pixels), dim=1).numpy()[:,y[active]]))
            
            # check what neighbor is best for current pixel
            for n in range(N_neighbors):
                # reset z\

                # calculate prediction for current neighbor
                with torch.no_grad():
                    temp = pixels[:,p,:].copy() # save original perturbation
                    pixels[:,p,:] = pixels_neighbors[:,p,n,:].copy()
                    z_neighbor = add_pixels(z_2d,pixels,active)
                    x_neighbor = self.adversarial_preprocess.inverse(z_neighbor).detach()
                    
                    pred_neighbor = np.diag(torch.nn.functional.softmax(self.model(x_neighbor), dim=1).numpy()[:,y[active]])
                    pixels[:,p,:] = temp.copy() # restore to original state
                
                if targeted == False:
                    idxs = np.nonzero(pred_neighbor <= pred_best[active])[0]
                else:
                    idxs = np.nonzero(pred_neighbor >= pred_best[active])[0]
                
                pred_best[active[idxs]] = np.squeeze(pred_neighbor[idxs].copy())
                pixels[active[idxs],p,:] = pixels_neighbors[active[idxs],p,n,:].copy() # swap original pixel for improvement
            
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
            print("----------------")
            print(pred_best)
            for b in range(N_BATCH):
                plt.scatter(i, pred_best[b], c=COLOR_LIST[int(20*b/N_BATCH)])
                plt.show()
                plt.pause(0.0001)
            plt.savefig('output\\Accuracy_LG_N32_'+str(int(eps))+str(targeted) + '.png')
    
        accuracy = np.hstack((accuracy, np.expand_dims(pred_best, -1)))
        

    with open('output\\Accuracy_LG_N32_'+str(int(eps))+str(targeted) + '.csv','ab+') as f:
        np.savetxt(f, accuracy, delimiter=",")

    
    z_adv = add_pixels(z_2d,pixels,np.arange(0,N_BATCH))
    z_adv = signal.resample(z_adv,z_shape[2],axis=2)
    
    self.adversarial_preprocess.forward(x)
    x_adv = self.adversarial_preprocess.inverse(z_adv).detach()
    

    with torch.no_grad(): 
        y_estimate_adversarial = torch.nn.functional.softmax(self.model(x_adv),dim=1) # DO THIS OUTSIDE OF THIS FUNCTION?
        y_estimate = torch.nn.functional.softmax(self.model(x_original.to(torch.float32)),dim=1)
    
    noise = x_adv - x_original
    
    self.model.train()
    return x_adv.to(torch.float32), noise.to(torch.float32), y_estimate_adversarial.to(torch.float32), y_estimate.to(torch.float32)
