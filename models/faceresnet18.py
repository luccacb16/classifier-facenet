import torch
import torch.nn as nn
from torchvision.models import resnet18

class FaceResNet18(nn.Module):
    def __init__(self, n_classes, emb_size):
        super(FaceResNet18, self).__init__()
        resnet = resnet18()
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add new layers
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, n_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        
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
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def get_embedding(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn1(x)
        return x