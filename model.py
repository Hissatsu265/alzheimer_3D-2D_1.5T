import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

class AlzheimerNet(nn.Module):
    def __init__(self, num_classes=3):  # 3 classes: AD, CN, MCI
        super(AlzheimerNet, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


