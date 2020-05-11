import torch
import numpy as np
import preprocess
import masking
import progressbar
import time


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
            elif adversarial_type == 'DE':
                x_adv = self.generate_adversarial_DE(x, target, targeted, x_min, x_max, train)
            elif adversarial_type == 'DE_masking':
                x_adv = self.generate_adversarial_DE_masking(x, target, targeted, x_min, x_max, train)
            elif adversarial_type == 'GL':
                x_adv = self.generate_adversarial_GL(x, target, targeted, x_min, x_max, train)
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
            elif adversarial_type == 'DE':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_DE_masking(x, target, targeted, 0, 90, train, mask=False)
            elif adversarial_type == 'DE_masking':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_DE_masking(x, target, targeted, 10, 20, train)
            elif adversarial_type == 'GL':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_GL_batch(x, target, targeted, 10, 20, train=train, mask=False)
            elif adversarial_type == 'GL_masking':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_GL_batch(x, target, targeted, 10, 20, train=train, mask=True)
            elif adversarial_type == 'free':
                print('Not yet implemented')
            elif adversarial_type == 'fast':
                print('Not yet implemente')
            return x_adv, x_delta, y_estimate_adv, y_estimate



    # VAC algorithm # put in 50 perturbations on random positions and do local optimization on each, over and over again.
    
    def generate_adversarial_GL_batch(self,x, y, targeted=False,x_min=0,x_max=1,train=False,N_perturbations=50, N_search=300, mask=True, intitalization_max=False):
        self.model.eval()
        x_original = x.clone().detach() # save for evaluation at the end
        z = self.adversarial_preprocess.forward(x)
        
        z_np = z.numpy()
        if mask == True:
            m = masking.get_mask_batches(x_original.numpy(), z_np, 16000, z_np.shape[2]) - 30
        else:
            m = np.ones_like(z_np) * (96-20)

        # get shapes
        shape = z_np.shape
        # should be (batch, data, ... )
        
        N_batch = shape[0]
        N_dimensions_position = len(shape) - 2 # subtract batch and data dim
        max_pos = shape[2:]

        N_neighbors = 5
        N_pixels = 100
        N_loops = 100

        def add_pixels(z, pixels):
            z_perturbed = z.copy()
            for b in range(N_batch):
                for i in range(N_pixels):
                    #import pdb; pdb.set_trace()
                    z_perturbed[b,0,pixels[b,i,0],pixels[b,i,1]] = m[b,0,pixels[b,i,0],pixels[b,i,1]]
            return z_perturbed
                



        # initialize pixels
        pixels = np.random.rand(N_batch, N_pixels, 2) # each pixel has two dimensions (x_pos, y_pos)
        pixels[:,:,0] = pixels[:,:,0] * (max_pos[0]-1)
        pixels[:,:,1] = pixels[:,:,1] * (max_pos[1]-1)
        pixels = pixels.astype(np.int)

        
        pixels_neighbors = np.zeros(pixels.shape[:-1] + (N_neighbors,) + (2,), dtype=np.int)
        
        
        # main optimization loops
        for i in range(N_loops):
            print(i)
            # create broadcasting map for this instead.
            pixels_neighbors[:,:,0,0] = pixels[:,:,0] + 1
            pixels_neighbors[:,:,0,1] = pixels[:,:,1]

            pixels_neighbors[:,:,1,0] = pixels[:,:,0] - 1
            pixels_neighbors[:,:,1,1] = pixels[:,:,1] 

            pixels_neighbors[:,:,2,0] = pixels[:,:,0]
            pixels_neighbors[:,:,2,1] = pixels[:,:,1] + 1 

            pixels_neighbors[:,:,3,0] = pixels[:,:,0]
            pixels_neighbors[:,:,3,1] = pixels[:,:,1] - 1 

            pixels_neighbors[:,:,4,0] = np.random.randint(0, max_pos[0])
            pixels_neighbors[:,:,4,1] = np.random.randint(0, max_pos[1])

            pixels_neighbors[pixels_neighbors[:,:,:,0] > max_pos[0]-1] = max_pos[0] - 1
            pixels_neighbors[pixels_neighbors[:,:,:,1] > max_pos[1]-1] = max_pos[1] - 1
            

            # optimize each pixel at a time
            for p in range(N_pixels):
                
                # calculate prediction with current pixel positions
                with torch.no_grad():
                    x_pixels = self.adversarial_preprocess.inverse(add_pixels(z_np,pixels)).detach()
                    pred_best = np.array(np.diag(torch.nn.functional.softmax(self.model(x_pixels), dim=1).numpy()[:,y]))
                    
                
                # check what neighbor is best for current pixel
                for n in range(N_neighbors):
                    # calculate prediction for current neighbor
                    with torch.no_grad():
                        
                        temp = pixels[:,p,:].copy() # save original
                        pixels[:,p,:] = pixels_neighbors[:,p,n,:].copy()
                        
                        x_neighbor = self.adversarial_preprocess.inverse(add_pixels(z_np,pixels)).detach()
                        pred_neighbor = np.diag(torch.nn.functional.softmax(self.model(x_neighbor), dim=1).numpy()[:,y])

                        pixels[:,p,:] = temp.copy() # restore to original state
                    
                    if targeted == False:
                        idxs = np.nonzero(pred_neighbor < pred_best)[0]
                    else:
                        idxs = np.nonzero(pred_neighbor > pred_best)[0]

                    
                    pred_best[idxs] = pred_neighbor[idxs].copy()
                    
                    pixels[idxs,p,:] = pixels_neighbors[idxs,p,n,:].copy() # swap original pixel for better neighbor
            print(pred_best[0:4])
                                
        
        x_adv = self.adversarial_preprocess.inverse(add_pixels(z_np,pixels)).detach()
        
        if train:
            self.model.train()
            return x_adv

        with torch.no_grad(): 
            y_estimate_adversarial = torch.nn.functional.softmax(self.model(x_adv),dim=1) # DO THIS OUTSIDE OF THIS FUNCTION?
            y_estimate = torch.nn.functional.softmax(self.model(x_original.to(torch.float32)),dim=1)
            #print(np.diag(y_estimate.numpy()[:,y]) - np.diag(y_estimate_adversarial.numpy()[:,y])) #print difference from original
            #print(np.diag(y_estimate_adversarial.numpy()[:,y])) # print probabilities adversarial

        
        noise = x_adv - x_original
        
        self.model.train()
        return x_adv.to(torch.float32), noise.to(torch.float32), y_estimate_adversarial.to(torch.float32), y_estimate.to(torch.float32)

            
                

    
    def generate_adversarial_GL(self, x, y, targeted=False,x_min=0,x_max=1,train=False,N_perturbations=50, N_search=300, mask=True, intitalization_max=False):
        self.model.eval()
        x_old = x.clone().detach() # save for evaluation at the end
        x = self.adversarial_preprocess.forward(x)
        
        x_np = x.numpy()
        if mask == True:
            m = masking.get_mask_batches(x_old.numpy(), x_np, 16000, x_np.shape[2]) - 30
        else:
            m = np.ones_like(x_np) * (96-30)

        # get shapes
        shape = x_np.shape
        input(shape)
        # should be (batch, data, ... )
        
        N_batch = shape[0]
        N_dimensions_position = len(shape) - 2 # subtract batch and data dim
        max_pos = shape[2:]


        def get_neighbors(px,py):
            n = []

            for idx_x in range(-1,2):
                for idx_y in range(-1,2):
                    n.append([px+idx_x, py+idx_y])
            for i in n:
                i[0] = max(i[0],0)
                i[0] = min(i[0],max_pos[0]-1)
                i[1] = max(i[1],0)
                i[1] = min(i[1],max_pos[1]-1)
            # add random candidate
            n.append([np.random.randint(0, max_pos[0]), np.random.randint(0, max_pos[1])])
            return n

        z_adv = x_np.copy()
        for b in range(N_batch):

            perturbations = []
            top_pred = 1
            print( "=============================================")
            for i in range(N_perturbations):
                tested_idxs = [] 

                
                # initialize all pixels to be close to max magnitude pixel 
                if intitalization_max == True:
                    pos_x, pos_y = np.unravel_index(x_np[b,0].argmax(), x_np[b,0].shape)
                    print("check that both comming values are ints")
                    input(pos_x)
                    input(pos_y)
                    pos_x += int(i/N_perturbations*np.random.rand() * max_pos[0])
                    pos_y += int(i/N_perturbations*np.random.rand() * max_pos[1])
                else:
                    pos_x = int(np.random.rand() * (max_pos[0]-1))
                    pos_y = int(np.random.rand() * (max_pos[1]-1))

                pos_x = min(max(pos_x, 0), max_pos[0]-1)
                pos_y = min(max(pos_y, 0), max_pos[1]-1)

                improvement = False

                for s in range(N_search):
                    counter = s
                    top_pos_neighbor = []
                    found_better = False
                    for n in get_neighbors(pos_x, pos_y):
                        
                        """
                        if n in tested_idxs:
                            continue
                        tested_idxs.append(n)
                        """
                        
                        # eval
                        temp = x_np[b,0, n[0], n[1]]
                        x_np[b,0, n[0], n[1]] = m[b,0,n[0],n[1]]
                        
                        x_perturbed = self.adversarial_preprocess.inverse(x_np).detach()
                        x_perturbed.unsqueeze(0)
                        with torch.no_grad():
                            pred = torch.nn.functional.softmax(self.model(x_perturbed), dim=1).numpy()[b,y[b]]
                        
                        if top_pred > pred:
                            improvement = True
                            top_pred = pred
                            top_pos_neighbor = [n[0], n[1]]
                            found_better = True
                    
                        x_np[b, 0, n[0], n[1]] = temp # reset x
                    
                    if found_better == True:
                        # move to new best idx
                        pos_x = top_pos_neighbor[0]
                        pos_y = top_pos_neighbor[1]
                    else:
                        break
                
                
                # add best perturbation
                if improvement:
                    x_np[b,0, pos_x, pos_y] = m[b,0,pos_x,pos_y]

                # early stopping
                print(counter)
                print(top_pred)
                if top_pred < 0.1:
                    break
                
            z_adv[b] = x_np[b]

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
        
        self.model.train()
        return x_adv.to(torch.float32), noise.to(torch.float32), y_estimate_adversarial.to(torch.float32), y_estimate.to(torch.float32)


                
    def generate_adversarial_DE_masking(self, x, y, targeted=False, x_min=0, x_max=1, train=False, N_perturbations=1, N_iterations=50, N_population=50, mask=True):
        self.model.eval()
        x_old = x.clone().detach() # save for evaluation at the end
        x = self.adversarial_preprocess.forward(x)
        
        x_np = x.numpy()
        if mask == True:
            m = masking.get_mask_batches(x_old.numpy(), x_np, 16000, x_np.shape[2])
        else:
            m = np.ones_like(x_np) * 96

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
        
        self.model.train()
        return x_adv.to(torch.float32), noise.to(torch.float32), y_estimate_adversarial.to(torch.float32), y_estimate.to(torch.float32)


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
        
        
