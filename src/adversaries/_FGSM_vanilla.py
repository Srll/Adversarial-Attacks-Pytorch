import torch
import numpy as np
import progressbar
import time

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
    