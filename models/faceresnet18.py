import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

class FaceResNet18(nn.Module):
    def __init__(self, n_classes, emb_size=1):
        super(FaceResNet18, self).__init__()
        resnet = resnet18()
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add new layers
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, n_classes)

        self._initialize_weights()
        
        self.n_classes = n_classes
        self.emb_size = emb_size

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
    
    def save_checkpoint(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
            
        model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in self.state_dict().items()}
        
        checkpoint = {
            'state_dict': model_state_dict,
            'n_classes': self.n_classes,
            'emb_size': self.emb_size
        }
        torch.save(checkpoint, os.path.join(path, filename))
        
    @staticmethod
    def load_checkpoint(path):
        checkpoint = torch.load(path)
        model = FaceResNet18(n_classes=checkpoint['n_classes'], emb_size=checkpoint['emb_size'])
        model.load_state_dict(checkpoint['state_dict'])
        return model