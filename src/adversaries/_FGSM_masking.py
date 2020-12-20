import torch
from inaudible import masking
from scipy import signal
import numpy as np
def generate_adversarial_FGSM_masking(self, x, y, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, train = False):
        
        def db2mag(c):
            return torch.pow(10, c/10)
        def mag2db(c):
            return 10 * torch.log10(c)
            
        z = torch.stft(x, 64, 32)
        z_phase = torch.atan2(z[:,:,:,0], z[:,:,:,1])
        z_spectrogram = torch.sqrt(z[:,:,:,0].pow(2) + z[:,:,:,1].pow(2))

        z_spectrogram = z_spectrogram + torch.finfo(float).eps
        z2 = 10 * torch.log10(z_spectrogram)
        
        max_save = torch.max(z2,dim=1,keepdim=True)[0]
        z3 = 96 + z2 - max_save


        m = masking.get_mask_batch(signal.resample(x.numpy(), int(x.shape[1] * (44100/16000)), axis=1))
        m_ = m[:,:int(np.ceil(m.shape[1]/2.75))]
        m__ = signal.resample(m_, z3.shape[1], axis=1)
        mask = signal.resample(m__, z3.shape[2], axis=2)
        mask_torch = torch.tensor(mask)

        z3_mag = db2mag(z3)
        z3_adv_mag = torch.autograd.Variable(z3_mag.data, requires_grad=True)
        
        z3_adv = mag2db(z3_adv_mag)
        # do all inverse transforms
        z2_adv = z3_adv - 96 + max_save
        z1_adv = torch.pow(10, z2_adv/10)
        
        z_real = z1_adv * torch.sin(z_phase)
        z_imag = z1_adv * torch.cos(z_phase)
        z_adv = torch.stack([z_real, z_imag],dim=-1)
        x_adv = torch.istft(z_adv, 64, 32)
        #import pdb; pdb.set_trace()
        y_estimate = self.model(x_adv)
        if targeted:
            loss = self.criterion(y_estimate,y) 
        else: 
            loss = -self.criterion(y_estimate,y)
        
        self.model.zero_grad()
        if z3_adv.grad is not None:
            z3_adv.grad.data.fill_(0.0)
        loss.backward()
        #U = z3_adv.shape[1]*z3_adv.shape[2]
        #U = 1402
        #U = 1270
        #U = 1015

        U = 10999
        #U = 1750
        
        
        z_grad = z3_adv_mag.grad
        z_grad_flat = torch.reshape(z_grad, (z_grad.shape[0],z_grad.shape[1]*z_grad.shape[2]))
        z_grad_flat_abs = torch.abs(z_grad_flat)
        z_grad_flat_sign = z_grad_flat.sign()
        sort_flat_abs, idxs_flat_abs = z_grad_flat_abs.sort()
        idxs = idxs_flat_abs[:,-U:]
        
        mask_torch_flat = torch.reshape(mask_torch, (z_grad.shape[0],z_grad.shape[1]*z_grad.shape[2]))
        noise_flat = torch.zeros_like(mask_torch_flat)
        for i in range(z_grad.shape[0]):
            for u in range(U):
                noise_flat[i,idxs[i,u]] = db2mag(mask_torch_flat[i,idxs[i,u]]) * z_grad_flat_sign[i,idxs[i,u]]
        
        noise = torch.reshape(noise_flat, (z_grad.shape[0],z_grad.shape[1],z_grad.shape[2]))
        z3_adv_noise = mag2db(db2mag(z3_adv) + eps*noise)
        #z3_adv = mag2db(db2mag(z3_adv))

        z2_adv = z3_adv_noise - 96 + max_save
        z1_adv = torch.pow(10, z2_adv/10)
        
        z_real = z1_adv * torch.sin(z_phase)
        z_imag = z1_adv * torch.cos(z_phase)
        z_adv = torch.stack([z_real, z_imag],dim=-1)
        x_adv = torch.istft(z_adv, 64, 32)
        x_adv = (x_adv.type(torch.int16)).type(torch.float32)
        
        if train:
            return x_adv 

        with torch.no_grad():
            y_estimate_adversarial = self.model(x_adv)

        return x_adv, x_adv - x, y_estimate_adversarial, y_estimate