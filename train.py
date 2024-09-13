import pandas as pd
from tqdm import tqdm
import os
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import wandb.wandb_torch

from models.faceresnet50 import FaceResNet50
from models.faceresnet18 import FaceResNet18

from utils import parse_args, transform, aug_transform, CustomDataset, evaluate, WarmUpCosineAnnealingLR, save_model_artifact

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
DTYPE = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        DTYPE = torch.float16
        
NUM_VAL_SAMPLES = 128
USING_WANDB = False
LAST_EPOCH = -1

model_map = {
    'faceresnet50': FaceResNet50,
    'faceresnet18': FaceResNet18
}

dataset_map = {
    'CASIA': 'casia-faces',
    'PINS': '105_classes_pins_dataset'
}
        
# --------------------------------------------------------------------------------------------------------
    
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler,
    accumulation_steps: int,
    epochs: int,
    dtype: torch.dtype,
    device: str,
    checkpoint_path: str
):
    
    start_epoch = LAST_EPOCH + 1
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        num_batches = len(train_loader) // accumulation_steps
        if LAST_EPOCH != -1:
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}", unit="batch")
        else:
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
            
            with autocast(dtype=dtype, device_type=device):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                
            running_loss += loss.item() * accumulation_steps
        
        progress_bar.close()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy, val_loss = evaluate(model, val_loader, criterion, dtype=dtype, device=device)
        
        if USING_WANDB:
            wandb.log({
                'epoch': epoch+1,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'accuracy': epoch_accuracy,
                'lr': optimizer.param_groups[0]['lr']
            })
            
        if LAST_EPOCH != -1:
            print(f"Epoch [{epoch}/{epochs}] | accuracy: {epoch_accuracy:.4f} | loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            model.save_checkpoint(checkpoint_path, f'epoch_{epoch}.pt')
            if USING_WANDB: save_model_artifact(checkpoint_path, epoch)
        else:
            print(f"Epoch [{epoch+1}/{epochs}] | accuracy: {epoch_accuracy:.4f} | loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            model.save_checkpoint(checkpoint_path, f'epoch_{epoch+1}.pt')
            if USING_WANDB: save_model_artifact(checkpoint_path, epoch+1)
            
        scheduler.step()
        
# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    
    model_name = args.model
    batch_size = args.batch_size
    accumulation = args.accumulation
    epochs = args.epochs
    emb_size = args.emb_size
    min_lr = args.min_lr
    max_lr = args.max_lr
    LAST_EPOCH = args.last_epoch
    warmup_epochs = args.warmup_epochs
    num_workers = args.num_workers
    DATA_PATH = args.data_path
    CHECKPOINT_PATH = args.checkpoint_path
    compile = args.compile
    USING_WANDB = args.wandb
    
    accumulation_steps = accumulation // batch_size
    
    config = {
        'model': model_name,
        'batch_size': batch_size,
        'accumulation': accumulation,
        'epochs': epochs,
        'num_workers': num_workers,
        'data_path': DATA_PATH,
        'checkpoint_path': CHECKPOINT_PATH,
        'compile': compile,
        'min_lr': min_lr,
        'max_lr': max_lr,
        'warmup_epochs': warmup_epochs,
    }

    if USING_WANDB:
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project='classifier-facenet', config=config)

    # ------
    
    # Dados
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    train_df['path'] = train_df['path'].apply(lambda x: os.path.join(DATA_PATH, 'casia-faces/', x))
    n_classes = train_df['id'].nunique()
    
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    test_df['path'] = test_df['path'].apply(lambda x: os.path.join(DATA_PATH, 'casia-faces/', x))
    
    # Selecionando um número fixo de amostras para validação
    test_df = test_df.sample(n=NUM_VAL_SAMPLES, random_state=42).reset_index(drop=True)
    
    # Datasets e Loaders
    train_dataset = CustomDataset(train_df, transform=aug_transform, dtype=DTYPE)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    val_dataset = CustomDataset(test_df, transform=transform, dtype=DTYPE)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Modelo
    if model_name.lower() in model_map:
        model = model_map[model_name.lower()](n_classes=n_classes, emb_size=emb_size).to(device)
    else:
        raise ValueError(f'Model {model_name} not found')
    
    if LAST_EPOCH != -1:
        print(f'\nResuming from epoch {LAST_EPOCH+1}')
        model = model_map[model_name.lower()].load_checkpoint(os.path.join(CHECKPOINT_PATH, f'epoch_{LAST_EPOCH}.pt')).to(device)
        
    if compile:
        model = torch.compile(model)
    
    # Scaler, Otimizador e Scheduler
    scaler = GradScaler()
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': max_lr, 'weight_decay': 1e-5, 'initial_lr': max_lr}
    ])
    
    if LAST_EPOCH != -1:
        scheduler = WarmUpCosineAnnealingLR(optimizer, epochs, warmup_epochs, min_lr, max_lr, LAST_EPOCH-1)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer, epochs, warmup_epochs, min_lr, max_lr, LAST_EPOCH)
        
    # -----
    
    print(f'\nModel: {model.__class__.__name__}')
    print(f'Device: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {DTYPE}')
    
    print(f'\nImagens: {len(train_df)} | Identidades: {n_classes} | imgs/classes: {len(train_df) / n_classes:.2f}\n')
    
    train(
        model              = model,
        train_loader       = train_loader,
        val_loader         = val_loader,
        criterion          = criterion,
        optimizer          = optimizer,
        scaler             = scaler,
        scheduler          = scheduler,
        accumulation_steps = accumulation_steps,
        epochs             = epochs,
        dtype              = DTYPE,
        device             = device,
        checkpoint_path    = CHECKPOINT_PATH
    )