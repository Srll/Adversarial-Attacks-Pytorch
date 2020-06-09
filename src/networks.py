import torch, torchvision
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import preprocess


# TODO create more general spectrogram classifier(DNN class)

# use http instead of https
from torchvision.models.shufflenetv2 import model_urls

model_urls['shufflenetv2_x0.5'] = model_urls['shufflenetv2_x0.5'].replace('https://', 'http://')


class audio_RNN(nn.Module):
    # based on paper: CONVOLUTIONAL RECURRENT NEURAL NETWORKS FOR MUSIC CLASSIFICATION
    # similar implementation: https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af
    def __init__(self, nr_features, nr_classes, input_length):
        super(audio_RNN, self).__init__()
        N_LAYERS = 3
        FILTER_LENGTH = 3
        LSTM_COUNT = 96
        HIDDEN_SIZE = 64
        HIDDEN_SIZE = 10
        CONV_FILTER_COUNT = 56
        self.conv_1 = nn.Conv1d(nr_features,int(nr_features/2), FILTER_LENGTH, 1)
        self.bn_1 = nn.BatchNorm1d(int(nr_features/2))
        self.pool_1 = nn.MaxPool1d(2)

        self.conv_2 = nn.Conv1d(int(nr_features/2),int(nr_features/4), FILTER_LENGTH, 1)
        self.bn_2 = nn.BatchNorm1d(int(nr_features/4))
        self.pool_2 = nn.MaxPool1d(2)
        """
        self.conv_3 = nn.Conv1d(CONV_FILTER_COUNT,CONV_FILTER_COUNT, FILTER_LENGTH, 1)
        self.bn_3 = nn.BatchNorm1d(CONV_FILTER_COUNT)
        self.pool_3 = nn.MaxPool1d(2)
        """
        self.LSTM = torch.nn.LSTM(input_size= 14,
                                hidden_size = HIDDEN_SIZE,
                                num_layers  = LSTM_COUNT,
                                batch_first = True)
        self.l_linear = nn.Linear(HIDDEN_SIZE * int(nr_features/4), nr_classes)

    def forward(self, x):
        
        x1 = self.pool_1(F.relu(self.bn_1(self.conv_1(x))))
        x2 = self.pool_2(F.relu(self.bn_2(self.conv_2(x1))))
        #x3 = self.pool_3(F.relu(self.bn_3(self.conv_3(x2))))
        #import pdb; pdb.set_trace()
        x, (hidden, cell) = self.LSTM(x2)
        S = x.shape
        x = x.reshape(S[0],S[1]*S[2])
        x = self.l_linear(x)
        return x

class audio_M3(nn.Module):
    # based on paper: VERY DEEP CONVOLUTIONAL NEURAL NETWORKS FOR RAW WAVEFORMS
    def __init__(self, nr_classes, input_length):
        super(audio_M3, self).__init__()
        self.conv_1 = nn.Conv1d(1, 256, 200, 8) # original paper uses 1, 256, 80, 4
        self.bn_1 = nn.BatchNorm1d(256)
        self.pool_1 = nn.MaxPool1d(4)

        size = int(((input_length - 200) / 8) / 4 + 1)

        self.conv_2 = nn.Conv1d(256, 256, 3)
        self.bn_2 = nn.BatchNorm1d(256)
        self.pool_2 = nn.MaxPool1d(4)
        
        size = int(((size - 3) / 4))

        self.avg_pool = nn.AvgPool1d(size)# TODO fix better way of setting size in network
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
    def __init__(self, nr_classes, input_length):
        super(audio_M5, self).__init__()
        self.conv_1 = nn.Conv1d(1, 128, 80, 4) # original paper uses 1, 128, 80, 4
        self.bn_1 = nn.BatchNorm1d(128)
        self.pool_1 = nn.MaxPool1d(4) # original paper uses 4
        
        size = int(((input_length - 80) / 4) / 4 + 1)

        self.conv_2 = nn.Conv1d(128, 128, 3) # original paper uses 128, 128, 3
        self.bn_2 = nn.BatchNorm1d(128)
        self.pool_2 = nn.MaxPool1d(4) # original paper uses 4
        
        size = int(((size - 3) / 4) + 1)

        self.conv_3 = nn.Conv1d(128, 256, 3)
        self.bn_3 = nn.BatchNorm1d(256)
        self.pool_3 = nn.MaxPool1d(4) # original paper uses 4
        
        size = int(((size - 3) / 4) + 1)

        self.conv_4 = nn.Conv1d(256, 512, 3)
        self.bn_4 = nn.BatchNorm1d(512)
        self.pool_4 = nn.MaxPool1d(4) # original paper uses 4
        
        size = int(((size - 3) / 4))
        
        self.avg_pool = nn.AvgPool1d(size) # TODO fix better way of setting size in network
        
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
        
        
        x = self.avg_pool(x)
        
        x = x.permute(0, 2, 1) 
        x = self.fc(x)
        x = torch.squeeze(x)
        
        return x


class audio_MJ(nn.Module):
    # raw audio 11 conv layer deep classifier 
    def __init__(self, classes, input_length):
        super(audio_MJ, self).__init__()
        self.conv = nn.Conv1d(1, 64, 10, 1)
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        size = int(((input_length - 10) / 2) + 1)

        temp = []
        for i in range(3):
            temp.append(nn.Conv1d(64, 64, 6))
            temp.append(nn.BatchNorm1d(64))
            temp.append(torch.nn.ReLU())
            temp.append(nn.MaxPool1d(3))
            size = int(((size - 6) / 3) + 1)

        self.stack = nn.Sequential(*temp)
        
        self.avg_pool = nn.AvgPool1d(size-1)
        self.fc = nn.Linear(64, classes)
        
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


class CNN(torch.nn.Module):
    def __init__(self, network_type, dataset_name, preprocess_sequence):

        super(CNN, self).__init__()
        self.network_type = network_type
        self.preprocess = preprocess.PreProcess(preprocess_sequence)
        self.GPU_enabled = False

        if dataset_name == 'speech':
            classes = 10
            input_length = 16384
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
        elif self.network_type == 'audio_M3':
            self.model = audio_M3(classes, input_length)
        elif self.network_type == 'audio_M5':
            self.model = audio_M5(classes, input_length)
        elif self.network_type == 'audio_MJ':
            self.model = audio_MJ(classes, input_length)
        elif self.network_type == 'audio_RNN':
            input_length = int(input_length/128) # conversion due to MFCC transform
            self.model = audio_RNN(257,classes,input_length)

    def GPU(self, enable):
        if enable:
            if torch.cuda.is_available():
                self.GPU_enabled = True
                #torch.backends.cudnn.benchmark = True
                self.model.cuda()
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
