import numpy as np
import masking
import torch
import preprocess

t_x = torch.ones((3,2,15,17))
t_y = torch.ones((3))
t_p = preprocess.PreProcess(['None'])



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
            for i, max_idx in enumerate(max_pos):
                
                c_pos[b,:,:,1+i] = np.maximum(np.minimum(x_pos[b,:,0,:,1+i] + F * (x_pos[b,:,1,:,1+i] - x_pos[b,:,2,:,1+i]), max_idx-1), 0)
            
            c_val[b] = np.maximum(np.minimum(x_val[b,:,0,:,:] + F * (x_val[b,:,1,:,:] - x_val[b,:,2,:,:]), x_max), x_min)

        
        return c_pos, c_val

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

    x_np = pre.forward(x).numpy()
    # get shapes
    shape = x_np.shape
    # should be (batch, data, ... )
    
    N_batch = shape[0]
    N_dimensions_position = len(shape) - 2 # subtract batch and data dim
    max_pos = shape[2:]
    
    

    # create perturbation population 
    K = N_perturbations
    B = shape[0]
    I = N_population
    data_dims = shape[1]
    p_pos = np.random.randint(0, min(max_pos),(B,I,K,data_dims+1))    # position values (int)
    p_val = np.random.uniform(x_min, x_max, (B,I,K,x.shape[1])) # pixel values (float)
    
    for b in range(B):
        p_pos[b,:,:,0] = b

    y = y.numpy().astype(int)
    

    for _ in range(N_iterations):
        c_pos, c_val = evolve(p_pos, p_val)
        
        for i in range(N_population): # loop over population (allows batch size evaluation)
            c_p = add_perturbation(x_np, c_pos, c_val,i) # get perturbed x by c
            p_p = add_perturbation(x_np, p_pos, p_val,i) # get perturbed x by p


            
            with torch.no_grad():
                c_p = self.adversarial_preprocess.inverse(c_p)
                p_p = self.adversarial_preprocess.inverse(p_p)
                y_new = softmax(self.model(c_p).numpy()/10)
                y_old = softmax(self.model(p_p).numpy()/10)
                #y_new = np.array([[0.2,0.3,0.6],[0.6,0.2,0.1]])
                #y_old = np.array([[0.21,0.22,0.69],[0.3,0.6,0.1]])
            
            
            
            
            if targeted == False:
                idxs = np.nonzero(np.diag(y_new[:,y]) < np.diag(y_old[:,y]))
            else:
                idxs = np.nonzero(np.diag(y_new[:,y]) > np.diag(y_old[:,y]))
                
            
            p_pos[idxs,i,:,:] = c_pos[idxs,i,:,:]
            p_val[idxs,i,:,:] = c_val[idxs,i,:,:]

    y_pred = np.zeros((B,I))

    import pdb; pdb.set_trace()
    for i in range(I): # loop through all candidates
        p_p = add_perturbation(x_np, p_pos, p_val,i)
        p_p = self.adversarial_preprocess.inverse(parents_p)
        with torch.no_grad():
            if targeted == False:
                y_pred[:,i] = np.diag(self.model(parents_p).numpy()[:,y])
            else:
                y_pred[:,i] = np.diag(self.model(parents_p).numpy()[:,y])

    idx = np.argmax(y_pred, axis=1)
    z_adv = add_perturbation_batch(x_np, p_pos, p_rgb, idx)
    x_adv = self.adversarial_preprocess.inverse(z_adv)
    

    if train:
        return x_adv

    with torch.no_grad(): 
        y_estimate_adversarial = self.model(x_adv) # DO THIS OUTSIDE OF THIS FUNCTION?
        y_estimate = self.model(x_old.to(torch.float32))
    noise = x_adv - x_old
    
    return x_adv.to(torch.float32), noise.to(torch.float32), y_estimate_adversarial.to(torch.float32), y_estimate.to(torch.float32)


differential_evolution(t_x, t_y, t_p, 2, 5, 20,10)