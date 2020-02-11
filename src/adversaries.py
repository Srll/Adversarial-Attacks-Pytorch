import torch

class AdversarialGenerator(object):

    def __init__(self, model, criterion):

        super(AdversarialGenerator,self).__init__()
        self.model = model 
        self.criterion = criterion 
    
    def generate_adversarial(self, adversarial_type, x, y, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, train = False):

        if adversarial_type == 'none':
            return x
        elif adversarial_type == 'FGSM':
            return self.generate_adversarial_FGSM(x, y, targeted, eps, x_min, x_max, train)
        elif adversarial_type == 'IFGSM':
            print('Not yer implemented')
        elif adversarial_type == 'free':
            print('Not yet implemented')
        elif adversarial_type == 'fast':
            print('Not yet implemented')
        
    def generate_adversarial_FGSM(self, x, y, targeted=False, eps = 0.03, x_min = 0.0, x_max = 1.0, train = False):

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
