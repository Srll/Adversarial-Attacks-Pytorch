from inaudible import preprocess, masking
import torch
import numpy as np

def generate_adversarial_ONE_PIXEL(self, x, y, targeted=False, x_min=-1, x_max=1, train=False, nr_of_pixels=1):
    # TODO accept sparsity as a parameter, ie how many "pixels" should be altered
    x_old = x
    x = self.adversarial_preprocess.forward(x)

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

    x_np = x.numpy()
    B = x.shape[0]                      # Batch size
    data_dims = len(x.shape) - 2        # subtract Batch and RGB dimensions
    max_idx = np.min(x.shape[2:]) - 1   # Only supports square area of perturbations
    I = 20                               # Iterations of algorithm
    
    parents_pos = np.random.randint(0,max_idx,(B,I,data_dims+1))    # position values (int)
    for b in range(B):
        parents_pos[b,:,0] = b

    parents_rgb = np.random.uniform(x_min, x_max, (B,I, x.shape[1])) # pixel values (float)

    for _ in range(20): # iterations of DE

        children_pos, children_rgb = evolve(parents_pos, parents_rgb)
        for i in range(I):
            
            children_p = add_perturbation(x_np, children_pos, children_rgb,i)
            parents_p = add_perturbation(x_np, parents_pos, parents_rgb,i)
            

            with torch.no_grad():
                children_p = self.adversarial_preprocess.inverse(children_p)
                parents_p = self.adversarial_preprocess.inverse(parents_p)
                y_new = softmax(self.model(children_p).numpy())
                y_old = softmax(self.model(parents_p).numpy())


            if targeted == False:
                idxs = np.nonzero(np.diag(y_new[:,y]) < np.diag(y_old[:,y]))
            else:
                idxs = np.nonzero(np.diag(y_new[:,y]) > np.diag(y_old[:,y]))

            if i == 0:
                #print(np.diag(y_old[:,y]))
                input(np.diag(y_old[:,y]))    
            
            parents_pos[idxs,i,:] = children_pos[idxs,i,:]
            parents_rgb[idxs,i,:] = children_rgb[idxs,i,:]

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
