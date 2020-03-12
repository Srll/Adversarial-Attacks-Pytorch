import torch
import numpy as np
import preprocess

class AdversarialGenerator(object):

    def __init__(self, model, criterion, preprocess_sequence):

        super(AdversarialGenerator,self).__init__()
        self.model = model
        self.criterion = criterion 
        self.adversarial_preprocess = preprocess.PreProcess(preprocess_sequence)
    
    def generate_adversarial(self, adversarial_type, x, target, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, alpha = 0.03, n_steps = 7, train = False, target_id=3):

        if train:
            if adversarial_type == 'none':
                return x
            elif adversarial_type == 'FGSM_vanilla':
                x_adv = self.generate_adversarial_FGSM_vanilla(x, target, targeted, eps, x_min, x_max, train)
            elif adversarial_type == 'PGD':
                x_adv = self.generate_adversarial_PGD(x, target, targeted, eps, x_min, x_max, alpha, n_steps, train)
            elif adversarial_type == 'ONE_PIXEL':
                x_adv = self.generate_adversarial_ONE_PIXEL(x, target, targeted, x_min, x_max, train)
            elif adversarial_type == 'free':
                print('Not yet implemented')
            elif adversarial_type == 'fast':
                print('Not yet implemented')
            return x_adv

        else: # evaluate
            if adversarial_type == 'none':
                return x, torch.zeros_like(x), self.model(x), self.model(x)
            elif adversarial_type == 'FGSM_vanilla':
                x_adv, x_delta, y_estimate_adv, y_estimate = self.generate_adversarial_FGSM_vanilla(x, target, targeted, eps, x_min, x_max, train)
            elif adversarial_type == 'PGD':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_PGD(x, target, targeted, eps, x_min, x_max, alpha, n_steps, train)
            elif adversarial_type == 'ONE_PIXEL':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_ONE_PIXEL(x, target, targeted, x_min, x_max, train)
            elif adversarial_type == 'free':
                print('Not yet implemented')
            elif adversarial_type == 'fast':
                print('Not yet implemente')
            return x_adv, x_delta, y_estimate_adv, y_estimate

    
    def generate_adversarial_ONE_PIXEL(self, x, y, targeted=False, x_min=-1, x_max=1, train=False, nr_of_pixels=1):
        # TODO add reduce computational complexity of algorithm, reduce amount of comparissons
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
            print(_)
            children_pos, children_rgb = evolve(parents_pos, parents_rgb)
            for i in range(I):
                
                children_p = add_perturbation(x_np, children_pos, children_rgb,i)
                parents_p = add_perturbation(x_np, parents_pos, parents_rgb,i)
                

                with torch.no_grad():
                    children_p = self.adversarial_preprocess.inverse(children_p)
                    parents_p = self.adversarial_preprocess.inverse(parents_p)
                    y_new = softmax(self.model(children_p).numpy()/100000)
                    y_old = softmax(self.model(parents_p).numpy()/100000)

                if targeted == False:
                    idxs = np.nonzero(np.diag(y_new[:,y]) < np.diag(y_old[:,y]))
                    #print(idxs)
                else:
                    idxs = np.nonzero(np.diag(y_new[:,y]) > np.diag(y_old[:,y]))
                    
                
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

    def generate_adversarial_FGSM_vanilla(self, x, y, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, train = False):
        
        x_adv = torch.autograd.Variable(x.data, requires_grad=True)
        y_estimate = self.model(x_adv)
        if targeted:
            loss = self.criterion(y_estimate,y) 
        else: 
            loss = - self.criterion(y_estimate,y)
        
        self.model.zero_grad() 
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0.0)
        loss.backward() 

        noise = eps * x_adv.grad.sign()
        x_adv = x_adv - noise
        x_adv = torch.clamp(x_adv,x_min,x_max)

        if train:
            return x_adv 

        with torch.no_grad():
            y_estimate_adversarial = self.model.model(x_adv)

        return x_adv, noise, y_estimate_adversarial, y_estimate
    
    def generate_adversarial_PGD(self, x, y, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, alpha = 0.001, n_steps = 7, train = False):

        delta = torch.rand_like(x) * (2.0 * eps) - eps # could also be initialized to 0
        for j in range(n_steps):
            delta = torch.autograd.Variable(delta, requires_grad = True)
            y_estimate = self.model.model(x + delta) 
            if targeted: 
                loss = self.criterion(y_estimate, y)
            else: 
                loss = - self.criterion(y_estimate, y)
            
            self.model.zero_grad() 
            if delta.grad is not None:
                delta.grad.data.fill_(0.0)
            loss.backward() 

            with torch.no_grad():
                delta = delta + alpha * delta.grad.sign() 
                delta = torch.max(torch.min(delta, torch.FloatTensor([eps])),torch.FloatTensor([-eps]))
        
        x_adv = x + delta  
        
        if train:
            return x_adv 
        
        
        with torch.no_grad(): 
            y_estimate = self.model.model(x)
            y_estimate_adversarial = self.model.model(x_adv) 
        
        
        return x_adv, delta, y_estimate_adversarial, y_estimate
        
        
