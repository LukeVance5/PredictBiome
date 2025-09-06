import torch.nn as nn
import torchvision.models as models


class ConvNeuralNet(nn.Module):
  def __init__(self, num_classes):
    super(ConvNeuralNet, self).__init__()
    self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = self.model.fc.in_features
    self.model.fc = nn.Linear(in_features, num_classes)
  def forward(self, x):
    return self.model(x)
