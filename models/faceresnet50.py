import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50
import warnings

class FaceResNet50(nn.Module):
    def __init__(self, n_classes=0, emb_size=256):
        super(FaceResNet50, self).__init__()
        resnet = resnet50()

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc1 = nn.Linear(2048, emb_size)
        self.bn1 = nn.BatchNorm1d(emb_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(emb_size, n_classes)

        self._initialize_weights()
        
        self.emb_size = emb_size
        self.n_classes = n_classes

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
        
    def load_checkpoint(path):
        # Suprimindo o FutureWarning específico para torch.load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(path)
    
        model = FaceResNet50(n_classes=checkpoint['n_classes'], emb_size=checkpoint['emb_size'])
        model.load_state_dict(checkpoint['state_dict'])
        return model