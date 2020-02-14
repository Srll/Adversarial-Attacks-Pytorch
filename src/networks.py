import torch, torchvision
import os

# use http instead of https
import torchvision.models
from torchvision.models.shufflenetv2 import model_urls

model_urls['shufflenetv2_x0.5'] = model_urls['shufflenetv2_x0.5'].replace('https://', 'http://')


class CNN(torch.nn.Module):

    def __init__(self, network_type, dataset_name):

        super(CNN, self).__init__() 
        self.network_type = network_type

        if dataset_name == 'speech':
            classes = 10
        elif dataset_name == 'dogscats':
            classes = 2 
        elif dataset_name == 'imagenet':
            classes = 6
        

        if self.network_type == 'images_resnet18':
            self.model = torchvision.models.resnet18()
            self.model.fc = torch.nn.Linear(512,classes)     
        elif self.network_type == 'images_mobilenetv2' :
            self.model = torchvision.models.mobilenet_v2()
            self.model.classifier = torch.nn.Linear(1280,classes)
        elif self.network_type == 'images_shufflenetv2':
            self.model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
            self.model.fc = torch.nn.Linear(1024,classes)  
    
    def forward(self, x):
        
        return self.model(x)