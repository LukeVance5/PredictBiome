import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle
from model.biome_cnn import ConvNeuralNet
dir = os.path.dirname(__file__)


def load_train_dataset_full():
  size = 128
  file_path = dir + "/means.pth"
  with open(file_path, 'rb') as file:
    loaded = pickle.load(file)
  all_transforms = transforms.Compose([transforms.Resize((size,size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=loaded["mean"],
                                                          std=loaded["std"])
                                     ])
  train_dataset_raw = datasets.ImageFolder(root=os.path.join(dir, '..', 'dataset/train'), transform = all_transforms)
  return train_dataset_raw

def load_test_dataset_full():
  size = 128
  file_path = dir + "/means.pth"
  with open(file_path, 'rb') as file:
    loaded = pickle.load(file)
  all_transforms = transforms.Compose([transforms.Resize((size,size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=loaded["mean"],
                                                          std=loaded["std"])
                                     ])
  test_dataset_raw = datasets.ImageFolder(root=os.path.join(dir, '..', 'dataset/test'), transform = all_transforms)
  return test_dataset_raw
  
def calculate_means():
  size = 128
  batch_size = 32
  all_transforms = transforms.Compose([transforms.Resize((size,size)),
                                     transforms.ToTensor()])
  train_dataset_raw = datasets.ImageFolder(root=os.path.join(dir, '..', 'dataset/train'), transform = all_transforms)
  loader = simple_dataloader(train_dataset_raw,batch_size)
  channels_sum, channels_squared_sum, num_batches = 0,0,0
  for data, _ in loader:
    channels_sum += torch.mean(data, dim=[0,2,3])
    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
    num_batches += 1
  mean = channels_sum / num_batches
  std = (channels_squared_sum/num_batches - mean**2)**0.5
  print(mean)
  print(std)  
  file_path = dir + "/means.pth"
  with open(file_path, 'wb') as file_handler:
    pickle.dump({"mean": mean, "std": std}, file_handler)

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
  file_path = dir + "/means.pth"
  with open(file_path, 'rb') as file:
    loaded = pickle.load(file)
  size = 128
  transform = transforms.Compose([transforms.Resize((size,size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean= loaded["mean"],
                                                          std= loaded["std"])
                                     ])
  processed = torch.unsqueeze(transform(image),0)
  return processed
