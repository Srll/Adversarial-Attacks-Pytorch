import numpy as np
import masking
import torch

def differential_evolution(x,y, pre,  N_dimensions_value, N_perturbations, N_iterations, N_population, targeted=False, x_min=-1, x_max=1, train=False):
    def evolve(p_pos, p_val, F=0.5):
        #p_pos = [B, N, K, 3]
        c_pos = np.copy(p_pos)
        c_val = np.copy(p_val)
        
        # pick out random individuals from population
        idxs = np.random.choice(I,size=(I,3)) # use same random process for whole batch
        x_pos = p_pos[:,idxs,:,:]
        x_val = p_val[:,idxs,:,:]
        
        for b in range(B):
            c_pos[b,:,:,1:] = np.maximum(np.minimum(x_pos[b,:,:,0,1:] + F * (x_pos[b,:,:,1,1:] - x_pos[b,:,:,2,1:]), max_idx), 0)
            c_val[b] = np.maximum(np.minimum(x_val[b,:,:,0,:] + F * (x_val[b,:,:,1,:] - x_val[b,:,:,2,:]), x_max), x_min)
        return c_pos, c_rgb

    def add_perturbation_batch(img, p_pos, p_val, idxs):
        img = img.copy()
        for b, i in enumerate(idxs.tolist()):
            for k in range(N_perturbations):
                img[p_pos[b,i,k,0], :, p_pos[b,i,k,1], p_pos[b,i,k,2]] = p_val[b,i,k,:]
        return torch.from_numpy(img).to(torch.float32)

    def add_perturbation(img0, p_pos, p_val, i):
        #img = [B, 3, max_x, max_y]
        #perturbation = [B, 400, 2]
        img = img0.copy()
        for k in range(N_perturbations):
            img[p_pos[:,i,k,0], :, p_pos[:,i,k,1], p_pos[:,i,k,2]] = p_val[:,i,k,:]
        img = torch.from_numpy(img)
        return img.to(torch.float32) # set type float32
    
    def softmax(x):
        return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T

    # get shapes
    shape = pre.forward(x).numpy().shape
    # should be (batch, data, ... )

    N_batch = shape[0]
    N_dimensions_position = len(shape) - 2 # subtract batch and data dim
    max_pos = shape[2:]
    
    
    # create perturbation population 
    K = N_perturbations
    p_pos = np.random.randint(0,max_idx,(B,I,K,data_dims+1))     # position values (int)
    p_val = np.random.uniform(x_min, x_max, (B,I,K,x.shape[1])) # pixel values (float)
    for b in range(B):
        p_pos[b,:,:,0] = b

    for _ in range(N_iterations):
        c_pos, c_val = evolve(p_pos, p_val)
        
        for i in range(N_population): # loop over population (allows batch size evaluation)
            c_p = add_perturbation(x_np, children_pos, children_rgb,i) # get perturbed x by c
            p_p = add_perturbation(x_np, parents_pos, parents_rgb,i) # get perturbed x by p

            with torch.no_grad():
                children_p = self.adversarial_preprocess.inverse(children_p)
                parents_p = self.adversarial_preprocess.inverse(parents_p)
                y_new = softmax(self.model(children_p).numpy()/10)
                y_old = softmax(self.model(parents_p).numpy()/10)


            if targeted == False:
                idxs = np.nonzero(np.diag(y_new[:,y]) < np.diag(y_old[:,y]))
            else:
                idxs = np.nonzero(np.diag(y_new[:,y]) > np.diag(y_old[:,y]))
                
            
            parents_pos[idxs,i,k,:] = children_pos[idxs,i,k,:]
            parents_rgb[idxs,i,k,:] = children_rgb[idxs,i,k,:]

    y_pred = np.zeros((B,I))

    for i in range(I): # loop thruogh all candidates
        parents_p = add_perturbation(x_np, parents_pos, parents_rgb,i)
        parents_p = self.adversarial_preprocess.inverse(parents_p)
        with torch.no_grad():
            if targeted == False:
                y_pred[:,i] = np.diag(self.model(parents_p).numpy()[:,y])
            else:
                y_pred[:,i] = np.diag(self.model(parents_p).numpy()[:,y])

    idx = np.argmax(y_pred, axis=1)
    z_adv = add_perturbation_batch(x_np, parents_pos, parents_rgb, idx)
    x_adv = self.adversarial_preprocess.inverse(z_adv)
    

    if train:
        return x_adv

    with torch.no_grad():
        
        y_estimate_adversarial = self.model(x_adv)
        y_estimate = self.model(x_old.to(torch.float32))
    noise = x_adv - x_old
    
    return x_adv.to(torch.float32), noise.to(torch.float32), y_estimate_adversarial.to(torch.float32), y_estimate.to(torch.float32)
