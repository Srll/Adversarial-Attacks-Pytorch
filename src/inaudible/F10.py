import torch
import torch.nn as nn
import torch.nn.functional as F

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
