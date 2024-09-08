import torch
import torch.nn as nn
from torchvision.models import resnet50

class FaceResNet50(nn.Module):
    def __init__(self, n_classes, emb_size=512):
        super(FaceResNet50, self).__init__()
        resnet = resnet50()

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc1 = nn.Linear(2048, emb_size)
        self.bn1 = nn.BatchNorm1d(emb_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(emb_size, n_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m is self.fc2:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)