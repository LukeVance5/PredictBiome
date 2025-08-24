import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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