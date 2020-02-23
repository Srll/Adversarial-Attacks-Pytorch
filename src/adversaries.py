import torch
import numpy as np

class AdversarialGenerator(object):

    def __init__(self, model, criterion):

        super(AdversarialGenerator,self).__init__()
        self.model = model 
        self.criterion = criterion 
    
    def generate_adversarial(self, adversarial_type, x, y, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, alpha = 0.03, n_steps = 7, train = False):

        if adversarial_type == 'none':
            return x
        elif adversarial_type == 'FGSM_vanilla':
            return self.generate_adversarial_FGSM_vanilla(x, y, targeted, eps, x_min, x_max, train)
        elif adversarial_type == 'PGD':
            return self.generate_adversarial_PGD(x, y, targeted, eps, x_min, x_max, alpha, n_steps, train)
        elif adversarial_type == 'ONE_PIXEL':
            return self.generate_adversarial_ONE_PIXEL(x, y, targeted)
        elif adversarial_type == 'free':
            print('Not yet implemented')
        elif adversarial_type == 'fast':
            print('Not yet implemented')
    

    
    def generate_adversarial_ONE_PIXEL(self, x_torch, y, targeted=False, train=False):
        x = x_torch.numpy()
        
        B = x.shape[0]
        max_x = x.shape[2]
        max_y = x.shape[3]
        I = 3
        pixel_res = 256

        def evolve(p, F=0.5):
            
            #p = [400, 5]
            

            c = np.copy(p)
            
            
            idxs = np.random.choice(p.shape[0],size=(400,3))
            x = p[idxs,:]
            
            for i in range (I):

                c[i,:] = x[i,0,:] + F * (x[i,1,:] - x[i,2,:])
                c[i,0] = np.minimum(np.maximum(c[i,0], 0), max_x-1)
                c[i,1] = np.minimum(np.maximum(c[i,1], 0), max_y-1)
                c[i,2] = np.minimum(np.maximum(c[i,2],0),1)
                c[i,3] = np.minimum(np.maximum(c[i,3],0),1)
                c[i,4] = np.minimum(np.maximum(c[i,4],0),1)
            return c

        def add_perturbation(img, perturbation):
            
            #img = [3, max_x, max_y]
            #perturbation = [5]
            
            img[:,int(perturbation[0]),int(perturbation[1])] = perturbation[2:]
            img = np.expand_dims(img, axis=0)
            
            img = torch.from_numpy(img)
            
            return img.to(torch.float32) # set type float32
        
        

        #parents[:,:,0] = (parents[:,:,0] * x.shape[1]) # scale and cast x_pixel_position to int in range [0,x_pix_max)
        #parents[:,:,1] = (parents[:,:,1] * x.shape[2]) # scale and cast y_pixel_position to int in range [0,y_pix_max)
        
        

        for b in range(B):
            # initalize population
            # generate uniform [0,1) in size (400, 3 or 5) depending on grey/RGB
            parents = np.random.uniform(0,1,(I, x.shape[1] + 2)) 
            print("b")
            x_img = x[b]
            for _ in range(2): # iterations of DE
                children = evolve(parents) # (400,5)
                
                for i in range(I):
                    print(train)
                    fitness_new = self.model(add_perturbation(x_img, children[i,:]))
                    fitness_old = self.model(add_perturbation(x_img, parents[i,:]))
                    
                    if targeted:
                        None
                    else:
                        if fitness_new[0,y] < fitness_old[0,y]:
                            parents[i,:] = children[i,:]
                     

            # save fittest for img
        
        
        x_adv = add_perturbation(x_img, parents[0,:])
        
        #if train:
        return x_adv
        
        
        y_estimate = self.model(x_torch)
        y_estimate_adversarial = self.model(x_adv)
        
        noise = x_adv - x_torch

        return x_adv, noise, y_estimate_adversarial, y_estimate
    

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
            y_estimate_adversarial = self.model(x_adv)

        return x_adv, noise, y_estimate_adversarial, y_estimate
    
    def generate_adversarial_PGD(self, x, y, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, alpha = 0.001, n_steps = 7, train = False):

        delta = torch.rand_like(x) * (2.0 * eps) - eps # could also be initialized to 0
        for j in range(n_steps):
            delta = torch.autograd.Variable(delta, requires_grad = True)
            y_estimate = self.model(x + delta) 
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
            y_estimate = self.model(x)
            y_estimate_adversarial = self.model(x_adv) 
        
        return x_adv, delta, y_estimate_adversarial, y_estimate
        
        
