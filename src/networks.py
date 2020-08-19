import torch, torchvision
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import preprocess


# TODO create more general spectrogram classifier(DNN class)

# use http instead of https
from torchvision.models.shufflenetv2 import model_urls

#torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_dtype(torch.float64)

model_urls['shufflenetv2_x0.5'] = model_urls['shufflenetv2_x0.5'].replace('https://', 'http://')



# Implementation https://github.com/fznsakib/environmental-sound-classifier/blob/master/train.py
class audio_conv2d_spectrogram(nn.Module):
    def __init__(self, nr_classes, input_shape):
        super(audio_conv2d_spectrogram, self).__init__()
        input_shape = (32,1, 257, 65)
        
        self.dropout = nn.Dropout(0.2)
        

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=32,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        self.batchNorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)


        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.batchNorm3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        
        self.batchNorm4 = nn.BatchNorm2d(64)

        # TODO automatically calculate size
        self.fc1 = nn.Linear(7168, 1024, bias=False)
        self.fc2 = nn.Linear(1024, nr_classes, bias=False)
        

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1, end_dim=3)

        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.dropout(x)
        x = self.fc2(x)
        return x

class audio_conv2d_mfcc(nn.Module):
    def __init__(self, nr_classes, input_shape):
        super(audio_conv2d_mfcc, self).__init__()
        input_shape = (32,1, 3*26, 65)
        
        self.dropout = nn.Dropout(0.2)
        

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        self.batchNorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)


        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.batchNorm3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        
        self.batchNorm4 = nn.BatchNorm2d(64)

        # TODO automatically calculate size
        self.fc1 = nn.Linear(5760, 1024, bias=False)
        self.fc2 = nn.Linear(1024, nr_classes, bias=False)
        

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1, end_dim=3)

        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.dropout(x)
        x = self.fc2(x)
        return x



class audio_waw2letter_mfcc(nn.Module):
    def __init__(self, nr_classes, input_length):
        super(audio_waw2letter_mfcc, self).__init__()
        #conv1 =  Cov1dBlock(input_size=input_size,output_size=256,kernal_size=(11,),stride=2,dilation=1,drop_out_prob=0.2,padding='same')

        self.layers = nn.Sequential(
            nn.Conv2d(26*3, 26*3, 8),
            torch.nn.ReLU(),
            nn.Conv1d(26*3, 100, 10),
            torch.nn.ReLU(),
            nn.Conv1d(100, 200, 20),
            torch.nn.ReLU(),
            nn.Conv1d(200, 400, 30),
            torch.nn.ReLU(),
        )
        
        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, nr_classes)

    def forward(self, x):
        x = self.layers(x)
        #import pdb; pdb.set_trace()
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.squeeze(x)
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

        self.conv_2 = nn.Conv1d(128, 128, 3) # original paper uses 128, 128, 3
        self.bn_2 = nn.BatchNorm1d(128)
        self.pool_2 = nn.MaxPool1d(4) # original paper uses 4

        self.conv_3 = nn.Conv1d(128, 256, 3)
        self.bn_3 = nn.BatchNorm1d(256)
        self.pool_3 = nn.MaxPool1d(4) # original paper uses 4

        self.conv_4 = nn.Conv1d(256, 512, 3)
        self.bn_4 = nn.BatchNorm1d(512)
        self.pool_4 = nn.MaxPool1d(4) # original paper uses 4
        
        self.avg_pool = nn.AvgPool1d(15) 
        
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


class audio_F7(nn.Module):
    # raw audio 11 conv layer deep classifier 
    def __init__(self, classes, input_length):
        super(audio_F7, self).__init__()
        self.conv = nn.Conv1d(1, 32, 10, 1)
        self.bn = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        
        

        network = []
        network.append(nn.Conv1d(32, 32, 5))
        network.append(nn.BatchNorm1d(32))
        network.append(torch.nn.ReLU())
        network.append(nn.MaxPool1d(3))

        network.append(nn.Conv1d(32, 32, 5))
        network.append(nn.BatchNorm1d(32))
        network.append(torch.nn.ReLU())
        network.append(nn.MaxPool1d(3))
        
        network.append(nn.Conv1d(32, 64, 5))
        network.append(nn.BatchNorm1d(64))
        network.append(torch.nn.ReLU())
        network.append(nn.MaxPool1d(3))
        
        network.append(nn.Conv1d(64, 64, 5))
        network.append(nn.BatchNorm1d(64))
        network.append(torch.nn.ReLU())
        network.append(nn.MaxPool1d(3))
        
        network.append(nn.Conv1d(64, 64, 5))
        network.append(nn.BatchNorm1d(64))
        network.append(torch.nn.ReLU())
        network.append(nn.MaxPool1d(3))
        
        network.append(nn.Conv1d(64, 128, 5))
        network.append(nn.BatchNorm1d(128))
        network.append(torch.nn.ReLU())
        network.append(nn.MaxPool1d(3))

        self.stack = nn.Sequential(*network)
        self.avg_pool = nn.AvgPool1d(9)
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
        x = torch.squeeze(x,1)
        
        return x

class audio_F10(nn.Module):
    # raw audio 10 conv layer deep classifier 
    def __init__(self, classes, input_length):
        super(audio_F10, self).__init__()
        conv_channels_1 = 32
        conv_channels_2 = 48
        conv_channels_3 = 96
        self.conv = nn.Conv1d(1, conv_channels_1, 10, 1)
        self.bn = nn.BatchNorm1d(conv_channels_1)
        self.pool = nn.MaxPool1d(3)
        
        size = int(((input_length - 10) / 3) + 1)

        network = []
        for i in range(3):
            
            network.append(nn.Conv1d(conv_channels_1, conv_channels_1, 3))
            network.append(nn.BatchNorm1d(conv_channels_1))
            network.append(torch.nn.ReLU())
            network.append(nn.MaxPool1d(2))
        
        network.append(nn.Conv1d(conv_channels_1, conv_channels_2, 3))
        network.append(nn.BatchNorm1d(conv_channels_2))
        network.append(torch.nn.ReLU())
        network.append(nn.MaxPool1d(2))
        for i in range(2):
            network.append(nn.Conv1d(conv_channels_2, conv_channels_2, 3))
            network.append(nn.BatchNorm1d(conv_channels_2))
            network.append(torch.nn.ReLU())
            network.append(nn.MaxPool1d(2))

        network.append(nn.Conv1d(conv_channels_2, conv_channels_3, 3))
        network.append(nn.BatchNorm1d(conv_channels_3))
        network.append(torch.nn.ReLU())
        network.append(nn.MaxPool1d(2))
        for i in range(2):
            network.append(nn.Conv1d(conv_channels_3, conv_channels_3, 3))
            network.append(nn.BatchNorm1d(conv_channels_3))
            network.append(torch.nn.ReLU())
            network.append(nn.MaxPool1d(2))
        
        self.stack = nn.Sequential(*network)
        self.avg_pool = nn.AvgPool1d(8)
        self.fc = nn.Linear(conv_channels_3, classes)
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv(x)
        x = F.relu(self.bn(x))
        x = self.pool(x)        
        x = self.stack(x)
        
        x = self.avg_pool(x)
        x = x.permute(0, 2, 1) 
        x = self.fc(x)
        x = torch.squeeze(x,1)
        
        return x



class CNN(torch.nn.Module):
    def __init__(self, network_type, dataset_name):

        super(CNN, self).__init__()
        self.network_type = network_type
        self.GPU_enabled = False
        self.preprocess_bool = False

        if dataset_name == 'speech':
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
        elif self.network_type == 'audio_M3':
            self.preprocess = preprocess.PreProcess(['cast_int16'])
            self.preprocess_bool = True
            self.model = audio_M3(classes, input_length)
        elif self.network_type == 'audio_M5':
            self.preprocess = preprocess.PreProcess(['cast_int16'])
            self.preprocess_bool = True
            self.model = audio_M5(classes, input_length)
        elif self.network_type == 'audio_MJ':
            self.preprocess = preprocess.PreProcess(['cast_int16'])
            self.preprocess_bool = True
            self.model = audio_MJ(classes, input_length)
        elif self.network_type == 'audio_F7':
            self.preprocess = preprocess.PreProcess(['cast_int16'])
            self.preprocess_bool = True
            self.model = audio_F7(classes, input_length)
        elif self.network_type == 'audio_F10':
            self.preprocess = preprocess.PreProcess(['cast_int16'])
            self.preprocess_bool = True
            self.model = audio_F10(classes, input_length)
        elif self.network_type == 'audio_conv2d_spectrogram':
            self.preprocess = preprocess.PreProcess(['spectrogram', 'insert_data_dim'])
            self.preprocess_bool = True
            self.model = audio_conv2d_spectrogram(classes, input_length)
        elif self.network_type == 'audio_conv2d_mfcc':
            self.preprocess = preprocess.PreProcess(['mfcc', 'insert_data_dim'])
            self.preprocess_bool = True
            self.model = audio_conv2d_mfcc(classes, input_length)
        #elif self.network_type == 'audio_RNN':
        #    input_length = int(input_length/128) # conversion due to MFCC transform
        #    self.model = audio_RNN(257,classes,input_length)
        

    def GPU(self, enable):
        if enable:
            if torch.cuda.is_available():
                self.GPU_enabled = True
                #torch.backends.cudnn.benchmark = True
                self.model.cuda()
                return
            else:
                print("No available CUDA device found, running on CPU instead")
        else:
            self.GPU_enabled = False
            self.model.cpu()
        
    def forward(self, x):
        if self.preprocess_bool == True:
            x = self.preprocess.forward(x)
        
        if self.GPU_enabled:
            return self.model(x.cuda()).cpu()
        else:
            return self.model(x)
