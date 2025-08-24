import torch
import os
from biome_cnn import ConvNeuralNet
from dataloader import load_test_dataset_full, simple_dataloader
path = os.path.dirname(__file__) + "/model.pth"
batch_size = 64
test_dataset = load_test_dataset_full()
test_loader = simple_dataloader(test_dataset, batch_size)
num_classes = len(test_dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvNeuralNet(num_classes) 
model.load_state_dict(torch.load(path))
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the {} train images: {} %'.format(len(test_dataset), round(100 * correct / total,2)))    