from logging import Logger
from data.augmentation import get_training_augmentation, get_validation_augmentation
from data.dataset import DentalDataset
from models.losses import ComboLoss
from models.unet import UNet
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm

from utils.metrics import SegmentationMetrics

def train(config):
    # Setup
    model = UNet(encoder=config.ENCODER, num_classes=len(config.CLASSES))
    model = model.to(config.DEVICE)
    
    criterion = ComboLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)
    
    scaler = GradScaler() if config.MIXED_PRECISION else None
    
    # Datasets
    train_dataset = DentalDataset(
        os.path.join(config.TRAIN_DIR, 'images'),
        os.path.join(config.TRAIN_DIR, 'masks'),
        transform=get_training_augmentation(config)
    )
    
    val_dataset = DentalDataset(
        os.path.join(config.VAL_DIR, 'images'),
        os.path.join(config.VAL_DIR, 'masks'),
        transform=get_validation_augmentation(config)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS)
    
    # Training loop
    logger = Logger(config.LOG_DIR)
    metrics = SegmentationMetrics(len(config.CLASSES))
    best_iou = 0.0
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer,
                                  scheduler, metrics, scaler, config)
        
        model.eval()
        val_metrics = validate(model, val_loader, criterion, metrics, config)
        
        # Logging
        logger.log_metrics(train_metrics, epoch, 'train')
        logger.log_metrics(val_metrics, epoch, 'val')
        
        # Save checkpoint if validation IoU improves
        mean_iou = val_metrics['mean_iou']
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
            }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))

def train_epoch(model, loader, criterion, optimizer, scheduler, metrics, scaler, config):
    epoch_metrics = {'loss': 0.0, 'mean_iou': 0.0}
    
    with tqdm(loader, desc='Training') as pbar:
        for images, masks in pbar:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if config.MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            batch_metrics = metrics.calculate_metrics(outputs, masks)
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['mean_iou'] += batch_metrics['mean_iou']
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{batch_metrics['mean_iou']:.4f}"
            })
        
        scheduler.step()
    
    # Average metrics over epoch
    num_batches = len(loader)
    epoch_metrics['loss'] /= num_batches
    epoch_metrics['mean_iou'] /= num_batches
    
    return epoch_metrics

def validate(model, loader, criterion, metrics, config):
    val_metrics = {'loss': 0.0, 'mean_iou': 0.0}
    
    with torch.no_grad():
        with tqdm(loader, desc='Validation') as pbar:
            for images, masks in pbar:
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                batch_metrics = metrics.calculate_metrics(outputs, masks)
                val_metrics['loss'] += loss.item()
                val_metrics['mean_iou'] += batch_metrics['mean_iou']
                
                pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_iou': f"{batch_metrics['mean_iou']:.4f}"
                })
    
    # Average metrics over validation set
    num_batches = len(loader)
    val_metrics['loss'] /= num_batches
    val_metrics['mean_iou'] /= num_batches
    
    return val_metrics
