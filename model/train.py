import torch
import torch.nn as nn
import os
from biome_cnn import ConvNeuralNet
from dataloader import load_train_dataset_full, simple_dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

train_dataset = load_train_dataset_full()
train_loader = simple_dataloader(train_dataset, batch_size)
num_classes = len(train_dataset.classes)
model = ConvNeuralNet(num_classes) 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  
path = os.path.dirname(__file__) + "/model.pth"

for epoch in range(num_epochs):
# Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))        
torch.save(model.state_dict(), path)    