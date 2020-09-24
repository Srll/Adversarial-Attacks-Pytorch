import torch
import numpy as np
from inaudible import preprocess, masking
import progressbar
from matplotlib import pyplot as plt
import time
from scipy import signal
import pickle
import os.path

class AdversarialGenerator(object):

    def __init__(self, model, criterion):

        super(AdversarialGenerator,self).__init__()
        self.model = model
        self.criterion = criterion 
        
    
    def generate_adversarial(self, adversarial_type, x, target, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, alpha = 0.03, n_steps = 7, train = False, target_id=3, verbose=False):
        if train:    
            if adversarial_type == 'none':
                return x
            elif adversarial_type == 'FGSM_vanilla':
                self.adversarial_preprocess = preprocess.PreProcess('none')
                x_adv = self.generate_adversarial_FGSM_vanilla(x, target, targeted, eps, x_min, x_max, train)
            elif adversarial_type == 'PGD':
                self.adversarial_preprocess = preprocess.PreProcess('none')
                x_adv = self.generate_adversarial_PGD(x, target, targeted, eps, x_min, x_max, alpha, n_steps, train)
            elif adversarial_type == 'ONE_PIXEL':
                self.adversarial_preprocess = preprocess.PreProcess('none')
                x_adv = self.generate_adversarial_ONE_PIXEL(x, target, targeted, x_min, x_max, train)
            elif adversarial_type == 'DE':
                self.adversarial_preprocess = preprocess.PreProcess(['resample_to_44100','spectrogram', 'insert_data_dim', 'mag2db96'])
                x_adv = self.generate_adversarial_DE(x, target, targeted, x_min, x_max, train)
            elif adversarial_type == 'DE_MASKING':
                print('DE_MASKING is not supported for adversarial training')
            elif adversarial_type == 'LGAP':
                print('LGAP is not supported for adversarial training')
            elif adversarial_type == 'RGAP':
                print('RGAP is not supported for adversarial training')
            elif adversarial_type == 'free':
                print('Not yet implemented')
            elif adversarial_type == 'fast':
                print('Not yet implemented')
            return x_adv

        else: # evaluate
            if adversarial_type == 'none':
                x.detach()
                with torch.no_grad(): 
                    y_estimate = torch.nn.functional.softmax(self.model(x),dim=1)   
                return x, torch.zeros_like(x), y_estimate,y_estimate
            elif adversarial_type == 'FGSM_vanilla':
                x_adv, x_delta, y_estimate_adv, y_estimate = self.generate_adversarial_FGSM_vanilla(x, target, targeted, eps, x_min, x_max, train)
            elif adversarial_type == 'PGD':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_PGD(x, target, targeted, eps, x_min, x_max, alpha, n_steps, train)
            elif adversarial_type == 'ONE_PIXEL':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_ONE_PIXEL(x, target, targeted, x_min, x_max, train)
            elif adversarial_type == 'DE':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_DE_MASKING(x, target, targeted, 0, 90, train, mask=False)
            elif adversarial_type == 'DE_MASKING':
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_DE_MASKING(x, target, targeted, 10, 20, train)
            elif adversarial_type == 'GL':
                self.adversarial_preprocess = preprocess.PreProcess(['resample_to_44100','spectrogram','insert_data_dim','mag2db96'])
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_GL_batch(x, target, targeted,eps, 0, 0, train=train, mask=False)
            elif adversarial_type == 'LGAP':
                self.adversarial_preprocess = preprocess.PreProcess(['resample_to_44100','spectrogram','insert_data_dim','mag2db96'])
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_LGAP(x, target, targeted,eps, 0,0,train=train, mask=True,verbose=verbose)
            elif adversarial_type == 'RGAP':
                self.adversarial_preprocess = preprocess.PreProcess(['resample_to_44100','spectrogram', 'insert_data_dim', 'mag2db96'])
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_RGAP(x, target, targeted, eps, 20, train=train,verbose=verbose)
            elif adversarial_type == 'brute_force_mask_reduce':
                self.adversarial_preprocess = preprocess.PreProcess(['resample_to_44100','spectrogram','insert_data_dim','mag2db96'])
                x_adv, x_delta, y_estimate_adv, y_estimate =  self.generate_adversarial_brute_force_reduce(x, target, targeted, 10, 20, train=train)
            elif adversarial_type == 'free':
                print('Not yet implemented')
            elif adversarial_type == 'fast':
                print('Not yet implemente')
            return x_adv, x_delta, y_estimate_adv, y_estimate

    
    # ADVERSARIAL ALGORTIHMS
    from ._RGAP import generate_adversarial_RGAP
    from ._LGAP import generate_adversarial_LGAP
    from ._ONE_PIXEL import generate_adversarial_ONE_PIXEL
    from ._FGSM_vanilla import generate_adversarial_FGSM_vanilla
    from ._PGD import generate_adversarial_PGD
    from ._DE_MASKING import generate_adversarial_DE_MASKING

