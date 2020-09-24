from inaudible import preprocess, masking
import torch
import numpy as np
import progressbar
from matplotlib import pyplot as plt
import time
from scipy import signal
import pickle
import os.path

def generate_adversarial_DE_MASKING(self, x, y, targeted=False, x_min=0, x_max=1, train=False, N_perturbations=2000, N_iterations=50, N_population=10, mask=True):
    self.model.eval()
    x_old = x.clone().detach() # save for evaluation at the end
    x = self.adversarial_preprocess.forward(x)
    
    x_np = x.numpy()
    if mask == True:
        m_32 = masking.get_mask_batch(signal.resample(x_original.numpy(), int(x.shape[1] * (FS_Z/FS_MODEL)), axis=1))
        
    m_2d = signal.resample(m_32,MAX_POS[1],axis=-1)     # resample to same time resolution
    m_2d = signal.resample(m_2d,F_RESOLUTION,axis=1)    # resample to same frequency resolution
    m_2d[m_2d > 96] = 96
    m_2d_mag = db2mag(m_2d)

    def evolve(p_pos, p_val, F=0.5):
        #p_pos = [B, N, K, 3]
        c_pos = np.copy(p_pos)
        c_val = np.copy(p_val)
        
        # pick out random individuals from population
        idxs = np.random.choice(I,size=(I,3)) # use same random process for whole batch
        x_pos = p_pos[:,idxs,:,:]
        x_val = p_val[:,idxs,:,:]
        
        
        for b in range(B):
            for i, max_idx in enumerate(max_pos):                    
                c_pos[b,:,:,1+i] = np.maximum(np.minimum(x_pos[b,:,0,:,1+i] + F * (x_pos[b,:,1,:,1+i] - x_pos[b,:,2,:,1+i]), max_idx-1), 0)
            c_val[b] = np.maximum(np.minimum(x_val[b,:,0,:,:] + F * (x_val[b,:,1,:,:] - x_val[b,:,2,:,:]), x_max), x_min)
        return c_pos, c_val

    def add_perturbation_batch(img, p_pos, p_val, idxs):
        img = img.copy()
        for b, i in enumerate(idxs.tolist()):
            for k in range(N_perturbations):
                for dim in range(data_dims):
                    img[p_pos[b,i,k,0], dim, p_pos[b,i,k,1], p_pos[b,i,k,2]] = np.maximum(m[p_pos[b,i,k,0],dim,p_pos[b,i,k,1],p_pos[b,i,k,2]] - p_val[b,i,k,dim], np.finfo(float).eps)
        return torch.from_numpy(img).to(torch.float32)

    def add_perturbation(img0, p_pos, p_val, i):
        #img = [B, 3, max_x, max_y]
        #perturbation = [B, 400, 2]
        img = img0.copy()
        for k in range(N_perturbations):
            for dim in range(data_dims):
                img[p_pos[:,i,k,0], dim, p_pos[:,i,k,1], p_pos[:,i,k,2]] = np.maximum(m[p_pos[:,i,k,0],dim,p_pos[:,i,k,1],p_pos[:,i,k,2]] - p_val[:,i,k,dim], np.finfo(float).eps)
        img = torch.from_numpy(img)
        return img.to(torch.float32) # set type float32
    
    # get shapes
    shape = x_np.shape
    print("---------------------")
    print(shape)
    print("---------------------")
    # should be (batch, data, ... )
    
    N_batch = shape[0]
    N_dimensions_position = len(shape) - 2 # subtract batch and data dim
    max_pos = shape[2:]

    # create perturbation population 
    K = N_perturbations
    B = shape[0]
    I = N_population
    data_dims = shape[1]
    p_pos = np.random.randint(0, min(max_pos),(B,I,K,N_dimensions_position+1))    # position values (int)
    p_val = np.random.uniform(x_min, x_max, (B,I,K,x.shape[1])) # pixel values (float)
    
    for b in range(B):
        p_pos[b,:,:,0] = b

    y = y.numpy().astype(int)        
    y_old = np.zeros((B,I))
    for i in range(N_population):
        p_p = add_perturbation(x_np, p_pos, p_val,i) # get perturbed x by p
        p_p_i = self.adversarial_preprocess.inverse(p_p)
        with torch.no_grad():
            y_old[:,i] = np.diag(torch.nn.functional.softmax(self.model(p_p_i), dim=1).numpy()[:,y])


    
    for _ in progressbar.progressbar(range(N_iterations), redirect_stdout=True):
        c_pos, c_val = evolve(p_pos, p_val)
        
        for i in range(N_population): # loop over population (allows batch size evaluation)
            c_p = add_perturbation(x_np, c_pos, c_val,i) # get perturbed x by c
            c_p_i = self.adversarial_preprocess.inverse(c_p)
            with torch.no_grad():
                y_new = np.diag(torch.nn.functional.softmax(self.model(c_p_i), dim=1).numpy()[:,y])
            
            if targeted == False:
                idxs = np.nonzero(y_new < y_old[:,i])[0]
            else:
                idxs = np.nonzero(y_new > y_old[:,i])[0]

            y_old[idxs,i] = y_new[idxs]
            p_pos[idxs,i,:,:] = c_pos[idxs,i,:,:]
            p_val[idxs,i,:,:] = c_val[idxs,i,:,:]
            
        
    
    y_pred = np.zeros((B,I))
    for i in range(I): # loop through all candidates
        p_p = add_perturbation(x_np, p_pos, p_val,i)
        p_p_i = self.adversarial_preprocess.inverse(p_p)
        with torch.no_grad():
            if targeted == False:
                y_pred[:,i] = np.diag(torch.nn.functional.softmax(self.model(p_p_i), dim=1).numpy()[:,y])
            else:
                y_pred[:,i] = np.diag(torch.nn.functional.softmax(self.model(p_p_i), dim=1).numpy()[:,y])

    idx = np.argmax(y_pred, axis=1)

    # calculate average distance from perturbation to max value
    f_distance = 0
    t_distance = 0
    magnitude = 0
    for i in range(N_batch):
        
        max_idx = np.unravel_index(x_np[i].argmax(), x_np[i].shape)
        f_distance += np.abs(max_idx[1] - p_pos[i][idx[i]][0][1])
        t_distance += np.abs(max_idx[2] - p_pos[i][idx[i]][0][2])
        magnitude += p_val[1][idx[1]][0][0]
    print("==============")
    print(f_distance/N_batch)
    print(t_distance/N_batch)
    print(magnitude/N_batch)

    
    z_adv = add_perturbation_batch(x_np, p_pos, p_val, idx)
    x_adv = self.adversarial_preprocess.inverse(z_adv)
    
    
    if train:
        self.model.train()
        return x_adv

    with torch.no_grad(): 
        y_estimate_adversarial = torch.nn.functional.softmax(self.model(x_adv),dim=1) # DO THIS OUTSIDE OF THIS FUNCTION?
        y_estimate = torch.nn.functional.softmax(self.model(x_old.to(torch.float32)),dim=1)
        #print(np.diag(y_estimate.numpy()[:,y]) - np.diag(y_estimate_adversarial.numpy()[:,y])) #print difference from original
        #print(np.diag(y_estimate_adversarial.numpy()[:,y])) # print probabilities adversarial

    noise = x_adv - x_old
    return x_adv.to(torch.float32), noise.to(torch.float32), y_estimate_adversarial.to(torch.float32), y_estimate.to(torch.float32)