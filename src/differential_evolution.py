import numpy as np
import masking

def differential_evolution(x,y, pre,  N_dimensions_value, N_perturbations, N_iterations, N_population, targeted=False, x_min=-1, x_max=1, train=False):

    def evolve(p_pos, p_rgb, F=0.5):
        #p_pos = [B, 400, 2]
        c_pos = np.copy(p_pos)
        c_rgb = np.copy(p_rgb)
        
        idxs = np.random.choice(I,size=(I,3)) # use same random process for whole batch
        x_pos = p_pos[:, idxs, :]
        x_rgb = p_rgb[:, idxs, :]
        
        for b in range(B):
            c_pos[b,:,1:] = np.maximum(np.minimum(x_pos[b,:,0,1:] + F * (x_pos[b,:,1,1:] - x_pos[b,:,2,1:]), max_idx), 0)
            c_rgb[b] = np.maximum(np.minimum(x_rgb[b,:,0,:] + F * (x_rgb[b,:,1,:] - x_rgb[b,:,2,:]), x_max), x_min)
        return c_pos, c_rgb

    def add_perturbation_batch(img, p_pos, p_rgb, idxs):
        img = img.copy()
        for b, i in enumerate(idxs.tolist()):
            img[p_pos[b,i,0], :, p_pos[b,i,1], p_pos[b,i,2]] = p_rgb[b,i,:]
        return torch.from_numpy(img).to(torch.float32)

    def add_perturbation(img0, p_pos, p_rgb, i):
        #img = [B, 3, max_x, max_y]
        #perturbation = [B, 400, 2]
        img = img0.copy()
        img[p_pos[:,i,0], :, p_pos[:,i,1], p_pos[:,i,2]] = p_rgb[:,i,:]
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

    p_pos = np.random.randint(0,max_idx,(B,I,data_dims+1))    # position values (int)
    p_val = np.random.uniform(x_min, x_max, (B,I, x.shape[1])) # pixel values (float)
    for b in range(B):
        parents_pos[b,:,0] = b


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


    
    


    np.zeros((N_population,))
    


    for i in range(N_iterations):  
        for j in N_population
    
    return