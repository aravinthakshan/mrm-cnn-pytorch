import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

def load_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = load_model('mnist_cnn.pth')
    predict(model, test_loader)
