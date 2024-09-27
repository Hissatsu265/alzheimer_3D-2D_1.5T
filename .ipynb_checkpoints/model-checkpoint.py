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

# Initialize the model and move to GPU
model = AlzheimerNet(num_classes=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model)  # Use multiple GPUs if available
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)