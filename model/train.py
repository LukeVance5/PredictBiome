import torch
import torch.nn as nn
from biome_cnn import ConvNeuralNet
from utils import load_train_dataset_full, simple_dataloader, save_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

train_dataset = load_train_dataset_full()
label_list = {value: key for key, value in train_dataset.class_to_idx.items()}
print(label_list)
train_loader = simple_dataloader(train_dataset, batch_size)
num_classes = len(train_dataset.classes)
model = ConvNeuralNet(num_classes) 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

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
save_model(model,num_classes,  label_list)      
  