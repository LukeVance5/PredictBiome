import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .biome_cnn import ConvNeuralNet
dir = os.path.dirname(__file__)

def load_train_dataset_full():
  size = 128
  all_transforms = transforms.Compose([transforms.Resize((size,size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5])
                                     ])
  train_dataset_raw = datasets.ImageFolder(root=os.path.join(dir, '..', 'dataset/train'), transform = all_transforms)
  return train_dataset_raw

def load_test_dataset_full():
  size = 128
  all_transforms = transforms.Compose([transforms.Resize((size,size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5])
                                     ])
  test_dataset_raw = datasets.ImageFolder(root=os.path.join(dir, '..', 'dataset/test'), transform = all_transforms)
  return test_dataset_raw
  
 
def simple_dataloader(dataset, batch_size):

  return torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

def save_model(model, num_classes, labels):
  print(labels)
  path = os.path.dirname(__file__) + "/model_struct.pth"
  torch.save({"model_state": model.state_dict(), "num_classes": num_classes, "labels": labels}, path)  
def load_model():
  path = os.path.dirname(__file__) + "/model_struct.pth"
  model_struct = torch.load(path)
  model = ConvNeuralNet(model_struct["num_classes"]) 
  model.load_state_dict(model_struct["model_state"])
  return model, model_struct["labels"]

def process_single_image(image):
  size = 128
  transform = transforms.Compose([transforms.Resize((size,size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5])
                                     ])
  processed = torch.unsqueeze(transform(image),0)
  return processed
