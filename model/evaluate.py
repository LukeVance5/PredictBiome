import torch
import torch.nn.functional as F
from model.biome_cnn import ConvNeuralNet
from model.utils import load_test_dataset_full, simple_dataloader, load_model

batch_size = 32
test_dataset = load_test_dataset_full()
test_loader = simple_dataloader(test_dataset, batch_size)
num_classes = len(test_dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model, _ = load_model()
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