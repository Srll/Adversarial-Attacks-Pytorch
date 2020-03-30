import torch, torchvision
import os
import torch.nn as nn
import torch.nn.functional as F
import preprocess


# TODO create more general spectrogram classifier(DNN class)

# use http instead of https
from torchvision.models.shufflenetv2 import model_urls

model_urls['shufflenetv2_x0.5'] = model_urls['shufflenetv2_x0.5'].replace('https://', 'http://')


class DNN(torch.nn.Module):
    def __init__(self, classes):
        super(DNN, self).__init__()
        self.conv1_1d = nn.Conv1d(in_channels=257, out_channels=128,kernel_size=16,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(2496, 200)
        self.fc2 = nn.Linear(200, classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1_1d(x)
        x = torch.unsqueeze(x, 1)

        x = self.pool(self.conv2(x))
        
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Simple_dense(torch.nn.Module):
    def __init__(self,classes):
        super(Simple_dense, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(48762, 3000)
        self.fc2 = nn.Linear(3000,800)
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
    def __init__(self, network_type, dataset_name, preprocess_sequence):

        super(CNN, self).__init__()
        self.network_type = network_type
        self.preprocess = preprocess.PreProcess(preprocess_sequence)
        

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
        elif self.network_type == 'audio_cdnn':
            self.model = DNN(classes)
            #self.model.fc = torch.nn.Linear(1024,classes)


        
    def forward(self, x):
        
        temp = self.preprocess.forward(x)
        return self.model(temp)

