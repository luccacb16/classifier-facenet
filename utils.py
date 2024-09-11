import argparse
from PIL import Image
import math

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import Dataset
from torch.amp import autocast
from torch.optim.lr_scheduler import _LRScheduler

transform = Compose([
    Resize((160, 160)),
    ToTensor(), 
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

aug_transform = Compose([
    RandomResizedCrop(160, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    RandomRotation(15),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, images_df, transform=None, dtype=torch.bfloat16):
        self.labels = images_df['id'].values
        self.image_paths = images_df['path'].values
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        image = image.to(self.dtype)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label
    
# --------------------------------------------------------------------------------------------------------

# EVAL

def evaluate(model, val_dataloader, criterion, dtype=torch.bfloat16, device='cuda'):
    correct = 0
    total = 0
    total_loss = 0.0
    
    model.eval()

    with autocast(dtype=dtype, device_type=device):
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device=device, dtype=dtype), labels.to(device=device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    model.train()
    
    accuracy = correct / total
    loss = total_loss / len(val_dataloader)

    return accuracy, loss

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, epochs, warmup_epochs, min_lr, max_lr, last_epoch=-1):
        self.epochs = epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.min_lr + (self.max_lr - self.min_lr) * alpha for _ in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_decay for _ in self.base_lrs]
    
# --------------------------------------------------------------------------------------------------------
    
def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a rede neural como classificador")
    parser.add_argument('--model', type=str, default='faceresnet50', help='Modelo a ser utilizado (default: faceresnet50)')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch (default: 128)')
    parser.add_argument('--accumulation', type=int, default=1024, help='Acumulação de gradientes (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, help='Número de epochs (default: 30)')
    parser.add_argument('--emb_size', type=int, default=256, help='Tamanho do embedding (default: 256)')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Taxa de aprendizado mínima (default: 1e-5)')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Taxa de aprendizado máxima (default: 1e-3)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Número de epochs para warmup (default: 5)')
    parser.add_argument('--num_workers', type=int, default=1, help='Número de workers para o DataLoader (default: 1)')
    parser.add_argument('--data_path', type=str, default='./data/', help='Caminho para o dataset (default: ./data/)')
    parser.add_argument('--dataset', type=str, default='CASIA', help='Dataset a ser utilizado (default: CASIA)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Caminho para salvar os checkpoints (default: ./checkpoints/)')
    parser.add_argument('--colab', type=bool, default=False, help='Se está rodando no Google Colab (default: False)')
    parser.add_argument('--wandb', type=bool, default=False, help='Se está rodando com o Weights & Biases (default: False)')
    
    return parser.parse_args()