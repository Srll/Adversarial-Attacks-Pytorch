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
