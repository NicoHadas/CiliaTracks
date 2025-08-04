import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN architecture
class TrajectoryCNN(nn.Module):
    """
    A Convolutional Neural Network for classifying cilia trajectory images.

    This network takes a grayscale image of particle trajectories as input
    and outputs a prediction score for two classes (e.g., 'Control' and 'PCD').
    The architecture consists of three convolutional blocks followed by a
    global average pooling layer and two fully connected layers.

    Attributes:
    -----------
    conv1 : nn.Sequential
        First convolutional block (Conv2d, ReLU, MaxPool2d).
    conv2 : nn.Sequential
        Second convolutional block.
    conv3 : nn.Sequential
        Third convolutional block.
    global_avg_pool : nn.AdaptiveAvgPool2d
        Global average pooling layer to reduce feature map dimensions.
    fc1 : nn.Linear
        First fully connected layer.
    fc2 : nn.Linear
        Output fully connected layer.
    """
    def __init__(self):
        """Initializes the layers of the TrajectoryCNN."""
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
        """
        Defines the forward pass of the TrajectoryCNN.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, height, width).

        Returns:
        --------
        torch.Tensor
            The output logits from the final fully connected layer.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x
