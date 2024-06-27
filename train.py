# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import CNN

# Custom DataLoader class
class MNISTDataLoader:
    def __init__(self, batch_size=64, val_split=0.1):
        # Transformations to be applied on the dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        # Load the dataset
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        # Split the dataset into training and validation sets
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create DataLoader for training and validation sets
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Load the test set
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training function
def train(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        evaluate(model, val_loader)

# Evaluation function
def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    data_loader = MNISTDataLoader()
    model = CNN()
    train(model, data_loader.train_loader, data_loader.val_loader)
