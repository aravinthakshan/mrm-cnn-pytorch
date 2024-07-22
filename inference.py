import torch
from torch.utils.data import DataLoader
from model import CNN
from train import MNISTDataLoader  # Import our custom data loader

# Load the trained model
def load_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to make predictions on the test set
def predict(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    # Create an instance of our custom data loader
    data_loader = MNISTDataLoader(batch_size=64)
    
    # Use the test_loader from our custom data loader
    test_loader = data_loader.test_loader

    # Load the trained model
    model = load_model('mnist_cnn.pth')

    # Make predictions
    predict(model, test_loader)