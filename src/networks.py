import torch, torchvision
import os
import torch.nn as nn
import torch.nn.functional as F
import preprocess


# TODO create more general spectrogram classifier(DNN class)

# use http instead of https
from torchvision.models.shufflenetv2 import model_urls

model_urls['shufflenetv2_x0.5'] = model_urls['shufflenetv2_x0.5'].replace('https://', 'http://')

    
class audio_M3(nn.Module):
    # based on paper: VERY DEEP CONVOLUTIONAL NEURAL NETWORKS FOR RAW WAVEFORMS
    def __init__(self, nr_classes):
        super(audio_M3, self).__init__()
        self.conv_1 = nn.Conv1d(1, 256, 80, 3)
        self.bn_1 = nn.BatchNorm1d(256)
        self.pool_1 = nn.MaxPool1d(4)
        self.conv_2 = nn.Conv1d(256, 256, 3)
        self.bn_2 = nn.BatchNorm1d(256)
        self.pool_2 = nn.MaxPool1d(4)
        self.avg_pool = nn.AvgPool1d(256)# TODO fix better way of setting size in network
        self.fc = nn.Linear(256, nr_classes)
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv_1(x)
        x = F.relu(self.bn_1(x))
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = F.relu(self.bn_2(x))
        x = self.pool_2(x)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = torch.squeeze(x)
        
        return x



class audio_M5(nn.Module):
    # based on paper: VERY DEEP CONVOLUTIONAL NEURAL NETWORKS FOR RAW WAVEFORMS
    def __init__(self, nr_classes):
        super(audio_M5, self).__init__()
        self.conv_1 = nn.Conv1d(1, 128, 80, 4) # original paper uses 1, 128, 80, 4
        self.bn_1 = nn.BatchNorm1d(128)
        self.pool_1 = nn.MaxPool1d(4) # original paper uses 4
        
        self.conv_2 = nn.Conv1d(128, 128, 3) # original paper uses 128, 128, 3
        self.bn_2 = nn.BatchNorm1d(128)
        self.pool_2 = nn.MaxPool1d(4) # original paper uses 4
        
        self.conv_3 = nn.Conv1d(128, 256, 3)
        self.bn_3 = nn.BatchNorm1d(256)
        self.pool_3 = nn.MaxPool1d(4) # original paper uses 4
        
        self.conv_4 = nn.Conv1d(256, 512, 3)
        self.bn_4 = nn.BatchNorm1d(512)
        self.pool_4 = nn.MaxPool1d(4) # original paper uses 4
        
         
        self.avg_pool = nn.AvgPool1d(1) # TODO fix better way of setting size in network
        
        self.fc = nn.Linear(512, nr_classes)
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv_1(x)
        x = F.relu(self.bn_1(x))
        x = self.pool_1(x)
        
        x = self.conv_2(x)
        x = F.relu(self.bn_2(x))
        x = self.pool_2(x)
        
        
        x = self.conv_3(x)
        x = F.relu(self.bn_3(x))
        x = self.pool_3(x)
        
        x = self.conv_4(x)
        x = F.relu(self.bn_4(x))
        x = self.pool_4(x)
        
        #print(x.size()) use to determine size for avg pool
        x = self.avg_pool(x)
        x = x.permute(0, 2, 1) 
        x = self.fc(x)
        x = torch.squeeze(x)
        
        return x


class audio_MJ(nn.Module):
    # raw audio 11 conv layer deep classifier 
    def __init__(self, classes):
        super(audio_MJ, self).__init__()
        self.conv = nn.Conv1d(1, 128, 10, 1)
        self.bn = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        
        temp = []
        for i in range(10):
            temp.append(nn.Conv1d(128, 128, 3))
            temp.append(nn.BatchNorm1d(128))
            temp.append(torch.nn.ReLU())
            temp.append(nn.MaxPool1d(2))

        self.stack = nn.Sequential(*temp)
        self.avg_pool = nn.AvgPool1d(5) 
        self.fc = nn.Linear(128, classes)
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv(x)
        x = F.relu(self.bn(x))
        x = self.pool(x)        
        
        x = self.stack(x)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 1) 
        x = self.fc(x)
        x = torch.squeeze(x)
        
        return x




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
"""
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
"""
class CNN(torch.nn.Module):
    def __init__(self, network_type, dataset_name, preprocess_sequence):

        super(CNN, self).__init__()
        self.network_type = network_type
        self.preprocess = preprocess.PreProcess(preprocess_sequence)
        self.GPU_enabled = False

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
        elif self.network_type == 'audio_M3':
            self.model = audio_M3(classes)
        elif self.network_type == 'audio_M5':
            self.model = audio_M5(classes)
        elif self.network_type == 'audio_MJ':
            self.model = audio_MJ(classes)


        """elif self.network_type == 'audio_conv_raw':
            self.model = TheModelClass(classes)
        elif self.network_type == 'simple_dense':
            self.model = Simple_dense(classes)"""

    def GPU(self, enable):
        if enable:
            if torch.cuda.is_available():
                self.GPU_enabled = True
                self.model.cuda()
                torch.backends.cudnn.benchmark = True
                return
            else:
                print("No available CUDA device, running on CPU")
        else:
            self.GPU_enabled = False
            self.model.cpu()
        
    def forward(self, x):
        x = self.preprocess.forward(x)
        if self.GPU_enabled:
            return self.model(x.cuda()).cpu()
        else:
            return self.model(x)
