import os
import pandas as pd
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import wandb.wandb_torch

from models.faceresnet50 import FaceResNet50
from models.faceresnet18 import FaceResNet18
from models.arcfaceresnet50 import ArcFaceResNet50

from utils import parse_args, transform, aug_transform, CustomDataset, evaluate, WarmUpCosineAnnealingLR, save_model_artifact

def setup(rank, world_size):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)

def cleanup(world_size):
    if world_size > 1:
        dist.destroy_process_group()

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

DTYPE = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)
    if gpu_properties.major < 8:
        DTYPE = torch.float16

NUM_VAL_SAMPLES = 128
USING_WANDB = False

model_map = {
    'faceresnet50': FaceResNet50,
    'faceresnet18': FaceResNet18,
    'arcface': ArcFaceResNet50
}

dataset_map = {
    'CASIA': 'casia-faces',
    'PINS': '105_classes_pins_dataset'
}

def train(
    rank,
    world_size,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    accumulation_steps,
    epochs,
    dtype,
    checkpoint_path
):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    for epoch in range(epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        num_batches = len(train_loader) // accumulation_steps
        if rank == 0:
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
            
            with autocast(dtype=dtype, device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs, labels)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if rank == 0:
                    progress_bar.update(1)
                
            running_loss += loss.item() * accumulation_steps
        
        if rank == 0:
            progress_bar.close()
        
        epoch_loss = running_loss / len(train_loader)
        if world_size > 1:
            dist.all_reduce(torch.tensor(epoch_loss).to(device), op=dist.ReduceOp.SUM)
            epoch_loss /= world_size
        
        if rank == 0:
            epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, val_loss = evaluate(model.module if world_size > 1 else model, val_loader, criterion, dtype=dtype, device=device)
            
            if USING_WANDB:
                wandb.log({
                    'epoch': epoch+1,
                    'train_loss': epoch_loss,
                    'val_loss': val_loss,
                    'accuracy': epoch_accuracy,
                    'precision': epoch_precision,
                    'recall': epoch_recall,
                    'f1': epoch_f1,
                    'lr': optimizer.param_groups[0]['lr']
                })
                
            print(f"Epoch [{epoch+1}/{epochs}] | loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Metrics: accuracy: {epoch_accuracy:.4f} | precision: {epoch_precision:.4f} | recall: {epoch_recall:.4f} | f1: {epoch_f1:.4f}\n")
            torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), os.path.join(checkpoint_path, f'epoch_{epoch+1}.pt'))
            if USING_WANDB: 
                save_model_artifact(checkpoint_path, epoch+1)
            
        scheduler.step()

def main(rank, world_size, args):
    setup(rank, world_size)
    
    model_name = args.model
    batch_size = args.batch_size // world_size if world_size > 0 else args.batch_size
    accumulation = args.accumulation
    epochs = args.epochs
    emb_size = args.emb_size
    min_lr = args.min_lr
    max_lr = args.max_lr
    warmup_epochs = args.warmup_epochs
    num_workers = args.num_workers // world_size if world_size > 0 else args.num_workers
    DATA_PATH = args.data_path
    CHECKPOINT_PATH = args.checkpoint_path
    compile = args.compile
    global USING_WANDB
    USING_WANDB = args.wandb
    random_state = args.random_state
    
    accumulation_steps = accumulation // batch_size
    
    if rank == 0:
        config = {
            'model': model_name,
            'batch_size': batch_size * world_size,
            'accumulation': accumulation,
            'epochs': epochs,
            'num_workers': num_workers * world_size,
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

    # Data
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    train_df['path'] = train_df['path'].apply(lambda x: os.path.join(DATA_PATH, 'casia-faces/', x))
    n_classes = train_df['id'].nunique()
    
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    test_df['path'] = test_df['path'].apply(lambda x: os.path.join(DATA_PATH, 'casia-faces/', x))
    
    # Selecting a fixed number of samples for validation
    test_df = test_df.sample(n=NUM_VAL_SAMPLES, random_state=random_state).reset_index(drop=True)
    
    # Datasets and Loaders
    train_dataset = CustomDataset(train_df, transform=aug_transform, dtype=DTYPE)
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    val_dataset = CustomDataset(test_df, transform=transform, dtype=DTYPE)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Model
    if model_name.lower() in model_map:
        model = model_map[model_name.lower()](n_classes=n_classes, emb_size=emb_size)
    else:
        raise ValueError(f'Model {model_name} not found')
    
    if compile:
        model = torch.compile(model)
    
    # Scaler, Optimizer and Scheduler
    scaler = GradScaler()
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': max_lr, 'weight_decay': 5e-4, 'initial_lr': max_lr}
    ])
    
    scheduler = WarmUpCosineAnnealingLR(optimizer, epochs, warmup_epochs, min_lr, max_lr, last_epoch=-1)
        
    if rank == 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'\nModel: {model.__class__.__name__}')
        print(f'Device: {device}')
        if torch.cuda.is_available():
            print(f'Number of GPUs: {torch.cuda.device_count()}')
            print(f'Device name: {torch.cuda.get_device_name(0)}')
        print(f'Using tensor type: {DTYPE}')
        print(f'\nImages: {len(train_df)} | Identities: {n_classes} | imgs/classes: {len(train_df) / n_classes:.2f}\n')
    
    train(
        rank               = rank,
        world_size         = world_size,
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
        checkpoint_path    = CHECKPOINT_PATH
    )
    
    cleanup(world_size)

if __name__ == '__main__':
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(0, world_size, args)