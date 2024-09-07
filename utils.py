import argparse
from PIL import Image
import os

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset
from torch.amp import autocast

transform = Compose(
    [
    Resize((160, 160)),
    ToTensor(), 
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

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
    
def save_checkpoint(model, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Salvar o estado do modelo removendo _orig_mod. do começo das chaves do dicionário -> torch.compile
    model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}   
    torch.save(model_state_dict, os.path.join(path, filename))            
    
# --------------------------------------------------------------------------------------------------------
    
def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a rede neural como classificador")
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch (default: 128)')
    parser.add_argument('--accumulation', type=int, default=1024, help='Acumulação de gradientes (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, help='Número de epochs (default: 30)')
    parser.add_argument('--emb_size', type=int, default=512, help='Tamanho do embedding (default: 512)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Taxa de aprendizado (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=1, help='Número de workers para o DataLoader (default: 1)')
    parser.add_argument('--data_path', type=str, default='./data/', help='Caminho para o dataset (default: ./data/)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Caminho para salvar os checkpoints (default: ./checkpoints/)')
    parser.add_argument('--colab', type=bool, default=False, help='Se está rodando no Google Colab (default: False)')
    parser.add_argument('--wandb', type=bool, default=False, help='Se está rodando com o Weights & Biases (default: False)')
    
    return parser.parse_args()