import torch, torchvision
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from inaudible import preprocess
from inaudible.F10 import audio_F10
from inaudible.F7 import audio_F7


from torchvision.models.shufflenetv2 import model_urls
model_urls['shufflenetv2_x0.5'] = model_urls['shufflenetv2_x0.5'].replace('https://', 'http://')

class CNN(torch.nn.Module):
    def __init__(self, network_type, dataset_name):

        super(CNN, self).__init__()
        self.network_type = network_type
        self.GPU_enabled = False
        self.preprocess_bool = False

        if 'speech' in dataset_name:
            classes = 10
            input_length = 16000
        elif dataset_name == 'dogscats':
            classes = 2 
        elif dataset_name == 'imagenet':
            classes = 6
        elif dataset_name == 'mnist':
            classes = 10       
        elif dataset_name == 'FMA_small':
            classes = 8
            input_length = int(2644992/12)
        
        if self.network_type == 'images_resnet18':
            self.model = torchvision.models.resnet18()
            self.model.fc = torch.nn.Linear(512,classes)     
        elif self.network_type == 'images_mobilenetv2' :
            self.model = torchvision.models.mobilenet_v2()
            self.model.classifier = torch.nn.Linear(1280,classes)
        elif self.network_type == 'images_shufflenetv2':
            self.model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
            self.model.fc = torch.nn.Linear(1024,classes)
        elif self.network_type == 'audio_F7':
            self.preprocess = preprocess.PreProcess(['cast_int16'])
            self.preprocess_bool = True
            self.model = audio_F7(classes, input_length)
        elif self.network_type == 'audio_F7_base':
            self.preprocess = preprocess.PreProcess(['cast_int16'])
            self.preprocess_bool = True
            self.model = audio_F7(classes, input_length)
        elif self.network_type == 'audio_F10':
            self.preprocess = preprocess.PreProcess(['cast_int16'])
            self.preprocess_bool = True
            self.model = audio_F10(classes, input_length)
        

    def GPU(self, enable):
        if enable:
            if torch.cuda.is_available():
                self.GPU_enabled = True
                self.model.cuda()
                return
            else:
                print("No available CUDA device found, running on CPU instead")
        self.GPU_enabled = False
        self.model.cpu()
        
    def forward(self, x):
        if self.preprocess_bool == True:
            x = self.preprocess.forward(x)
        
        if self.GPU_enabled:
            return self.model(x.cuda()).cpu()
        else:
            return self.model(x)
