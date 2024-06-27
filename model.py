# model.py
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: input channels = 1 (grayscale), output channels = 32, kernel size = 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: input channels = 32, output channels = 64, kernel size = 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Fully connected layer: input features = 64*7*7 (from the conv layers), output features = 128
        self.fc1 = nn.Linear(64*7*7, 128)
        # Output layer: input features = 128, output features = 10 (number of classes)
        self.fc2 = nn.Linear(128, 10)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # Apply second convolutional layer followed by ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64*7*7)
        # Apply dropout
        x = self.dropout(x)
        # Apply first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        # Apply output layer
        x = self.fc2(x)
        return x
