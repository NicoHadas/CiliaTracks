import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN architecture
class TrajectoryCNN(nn.Module):
    def __init__(self):
        super(TrajectoryCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (16, 500, 500)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (16, 250, 250)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (32, 250, 250)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 125, 125)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (64, 125, 125)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 62, 62)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, 128)  
        self.fc2 = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x
