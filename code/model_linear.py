import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, num_classes):
        '''
        Define the layers of the network

        Args:
        num_classes: int, number of classes in the dataset
        
        Returns:
        None
        '''
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 500)  # Input size is 3 * 32 * 32 for CIFAR-100
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 200)
        self.dropout = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.dropout = nn.Dropout(p=0.3)
        self.fc6 = nn.Linear(200, num_classes)

    def forward(self, x):
        '''
        Forward pass through the network
        
        Args:
        x: torch.tensor, input tensor
        
        Returns:
        x: torch.tensor, output tensor
        '''
        x = x.view(-1, 3 * 32 * 32)  # Reshape input to a vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x