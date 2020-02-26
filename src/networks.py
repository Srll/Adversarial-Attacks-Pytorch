import torch, torchvision
import os
import torch.nn as nn
import torch.nn.functional as F


# use http instead of https
from torchvision.models.shufflenetv2 import model_urls

model_urls['shufflenetv2_x0.5'] = model_urls['shufflenetv2_x0.5'].replace('https://', 'http://')

# SIMPLE SPECTORGRAM CONV NN
class spectro_conv(nn.Module):
    def __init__(self,classes):
        super(spectro_conv, self).__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        

# Define model
class audio_conv(nn.Module):
    def __init__(self, classes):
        super(audio_conv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, stride=4, kernel_size=500)
        self.pool1 = nn.MaxPool1d(4,4) # check this 4, 1 or 1, 4
        #self.bn1 = nn.BatchNorm1d(num_features=320)
        self.conv2 = nn.Conv1d(256,256,kernel_size=3)
        self.pool2 = nn.MaxPool2d(4, 4) # check this 4, 1 or 1, 4
        self.avgPool = nn.AvgPool1d(4)
        self.softMax = nn.Softmax(dim=-1)
        
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(7424, 10)
        

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = self.softMax(x)
        
        return x


class Simple_dense(torch.nn.Module):
    def __init__(self,classes):
        super(Simple_dense, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000,800)
        self.fc3 = nn.Linear(800,512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128,classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        

        return x

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
        elif dataset_name == 'mnist':
            classes = 10        
        

        if self.network_type == 'images_resnet18':
            self.model = torchvision.models.resnet18()
            self.model.fc = torch.nn.Linear(512,classes)     
        elif self.network_type == 'images_mobilenetv2' :
            self.model = torchvision.models.mobilenet_v2()
            self.model.classifier = torch.nn.Linear(1280,classes)
        elif self.network_type == 'images_shufflenetv2':
            self.model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
            self.model.fc = torch.nn.Linear(1024,classes)
        elif self.network_type == 'audio_conv_raw':
            self.model = TheModelClass(classes)
        elif self.network_type == 'simple_dense':
            self.model = Simple_dense(classes)
            
    
    def forward(self, x):
        
        return self.model(x)