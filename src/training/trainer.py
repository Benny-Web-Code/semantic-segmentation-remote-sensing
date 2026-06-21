import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = self._setup_loss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        self.best_loss = float('inf')
        self.patience = 0
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 15)
        
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Trainer initialized")

    def _setup_loss(self):
        from .losses import DiceLoss, FocalLoss
        loss_type = self.config['training'].get('loss_function', 'dice')
        
        if loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'focal':
            return FocalLoss()
        else:
            return nn.CrossEntropyLoss()

    def _setup_optimizer(self):
        optimizer_type = self.config['training'].get('optimizer', 'adam')
        lr = self.config['training'].get('learning_rate', 0.001)
        wd = self.config['training'].get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            momentum = self.config['training'].get('momentum', 0.9)
            return SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    def _setup_scheduler(self):
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        epochs = self.config['training'].get('epochs', 100)
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            return StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            rgb = batch['rgb'].to(self.device)
            ms = batch['multispectral'].to(self.device)
            sar = batch['sar'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(rgb, ms, sar)
            loss = self.criterion(outputs, mask)
            
            loss.backward()
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
            
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                rgb = batch['rgb'].to(self.device)
                ms = batch['multispectral'].to(self.device)
                sar = batch['sar'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                outputs = self.model(rgb, ms, sar)
                loss = self.criterion(outputs, mask)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch, name='best'):
        path = self.checkpoint_dir / f"{name}_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def train(self):
        epochs = self.config['training'].get('epochs', 100)
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience = 0
                self._save_checkpoint(epoch, 'best')
            else:
                self.patience += 1
            
            if self.patience >= self.early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        logger.info("Training completed")
