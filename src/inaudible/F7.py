import torch
import torch.nn as nn
import torch.nn.functional as F

class audio_F7(nn.Module):
    # raw audio 7 conv layer deep classifier 
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
