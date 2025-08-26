import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
from datetime import datetime
import json

# GeoNeXt imports
from models import geonext_tiny
from utils import h5_Dataset, evaluator, ModelEMA
from utils.train import training
from utils.evaluator import validationing
from utils.augmentations import create_augmentation_transforms


class TrainingExperiment:
    """Minimal training experiment class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.get("gpu_id", 0)}' if config.get('gpu_id', 0) != -1 and torch.cuda.is_available() else 'cpu')
        
        # Create experiment name and directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"GeoNeXt_{config['dataset']}_{config.get('model_variant', 'tiny')}_{timestamp}"
        if config.get('name_suffix'):
            base_name = f"{base_name}_{config['name_suffix']}"
        self.exp_name = base_name
        self.exp_dir = os.path.join('experiments', self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Training tracking
        self.train_history = []
        self.val_history = []
        self.best_val_miou = 0.0
        self.best_epoch = 0
    
    def _setup_model(self):
        """Setup model with pretrained weights support"""
        args = argparse.Namespace(**self.config)
        args.device = self.device
        
        # Validate ConvNeXtV2 pretrained weights
        if self.config.get('convnextv2_pretrained_weights'):
            if not os.path.exists(self.config['convnextv2_pretrained_weights']):
                print(f"WARNING: ConvNeXtV2 weights not found: {self.config['convnextv2_pretrained_weights']}")
                args.convnextv2_pretrained_weights = None
            else:
                print(f"Using ConvNeXtV2 weights: {self.config['convnextv2_pretrained_weights']}")
        
        # Create model
        model = geonext_tiny(args).to(self.device)
        
        # Load pretrained model if specified
        start_epoch = 0
        if self.config.get('pretrained_model_path'):
            start_epoch = self._load_pretrained_model(model, args)
        
        return model, start_epoch
    
    def _load_pretrained_model(self, model, args):
        """Load pretrained model weights and return starting epoch"""
        path = self.config['pretrained_model_path']
        hard_reset = self.config.get('hard_reset', False)
        
        if not os.path.exists(path):
            print(f"Pretrained model not found: {path}")
            return 0
        
        print(f"Loading pretrained model: {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            
            # Handle encoder-only loading
            if getattr(args, 'load_encoder_only', False):
                print("Loading encoder weights only")
                encoder_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder.')}
                if encoder_dict:
                    model.load_state_dict(encoder_dict, strict=False)
                    print(f"Loaded {len(encoder_dict)} encoder parameters")
                return 0
            else:
                # Load full model
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)} keys")
                print("Model weights loaded")
            
            # Handle hard reset vs normal resume
            if hard_reset:
                start_epoch = self.config.get('start_epoch', 0)
                print(f"Hard reset: starting from epoch {start_epoch}")
                return start_epoch
            else:
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming from epoch {start_epoch}")
                    return start_epoch
                return 0
                    
        except Exception as e:
            print(f"Error loading model: {e}")
            return 0
    
    def _setup_data_loaders(self):
        """Setup data loaders"""
        data_path = os.path.join('dataset', self.config['dataset'])
        batch_size = self.config.get('batch_size', 8)
        
        # Setup augmentation
        aug_transform = None
        if self.config.get('use_augmentation', False):
            aug_transform = create_augmentation_transforms(self.config['augmentation'])
        
        # Create datasets and loaders
        train_dataset = h5_Dataset(data_path, train=True, augmentation_transform=aug_transform)
        val_dataset = h5_Dataset(data_path, train=False, augmentation_transform=None)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, val_loader
    
    def _setup_optimizer_and_scheduler(self, model, train_loader):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 3e-4)
        wd = self.config.get('weight_decay', 1e-5)
        
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        # Scheduler
        scheduler_name = self.config.get('scheduler', 'none').lower()
        scheduler = None
        
        if scheduler_name == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.config.get('max_lr', 3.5e-3),
                epochs=self.config.get('epochs', 60), steps_per_epoch=len(train_loader),
                div_factor=self.config.get('div_factor', 65),
                final_div_factor=self.config.get('final_div_factor', 750),
                pct_start=self.config.get('pct_start', 0.1),
                anneal_strategy=self.config.get('anneal_strategy', 'cos')
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, 
                step_size=self.config.get('step_size', 20), gamma=self.config.get('step_gamma', 0.1))
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.get('epochs', 60))
        
        return optimizer, scheduler
    
    def run(self):
        """Run the training experiment"""
        # Set random seed
        if 'seed' in self.config:
            torch.manual_seed(self.config['seed'])
            np.random.seed(self.config['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config['seed'])
        
        print(f"Starting: {self.exp_name}")
        print(f"Directory: {self.exp_dir}")
        print(f"Device: {self.device}")
        
        # Setup components
        model, start_epoch = self._setup_model()
        train_loader, val_loader = self._setup_data_loaders()
        optimizer, scheduler = self._setup_optimizer_and_scheduler(model, train_loader)
        
        # Setup EMA
        ema = ModelEMA(model, decay=self.config.get('ema_decay', 0.9999)) if self.config.get('use_ema', False) else None
        
        # Setup evaluators and args
        train_evaluator, val_evaluator = evaluator(2), evaluator(2)
        args = argparse.Namespace(**self.config, device=self.device, ema=ema)
        
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
        
        # Validate configuration
        epochs = self.config.get('epochs', 60)
        if start_epoch >= epochs:
            print(f"Start epoch {start_epoch} >= total epochs {epochs}")
            return None
        
        print(f"Training epochs {start_epoch + 1} to {epochs}")
        print("=" * 80)
        
        # Training loop
        total_start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            print(f"\n Epoch {epoch+1}/{epochs}")
            epoch_start = time.time()
            
            # Training and validation
            train_results = training(train_loader, model, optimizer, epoch, train_evaluator, args)
            if scheduler and self.config.get('scheduler', 'none').lower() != 'onecycle':
                scheduler.step()
            val_results = validationing(val_loader, model, epoch, val_evaluator, args)
            
            # Track history
            self.train_history.append(train_results)
            self.val_history.append(val_results)
            
            # Progress metrics
            duration = time.time() - epoch_start
            total_time = time.time() - total_start_time
            lr = optimizer.param_groups[0]['lr']
            
            # ETA calculation
            epochs_done = epoch - start_epoch + 1
            avg_time = total_time / epochs_done
            eta_min = (avg_time * (epochs - epoch - 1)) / 60
            
            # F1 calculation (same as original)
            train_f1_raw = train_results.get('f1', [0, 0])
            val_f1_raw = val_results.get('f1', [0, 0])
            train_f1 = float((float(train_f1_raw[0]) + float(train_f1_raw[1])) / 2.0) * 100
            val_f1 = float((float(val_f1_raw[0]) + float(val_f1_raw[1])) / 2.0) * 100
            
            print(f" {duration:.1f}s | Total: {total_time/60:.1f}m | ETA: {eta_min:.1f}m | LR: {lr:.2e}")
            print(f"   Train - Loss: {train_results['loss']:.4f} | mIoU: {train_results['mIou']*100:.2f}% | F1: {train_f1:.2f}% | Acc: {train_results['accuracy']*100:.2f}%")
            print(f"   Val   - Loss: {val_results['loss']:.4f} | mIoU: {val_results['mIou']*100:.2f}% | F1: {val_f1:.2f}% | Acc: {val_results['accuracy']*100:.2f}%")
            
            # Progress bar
            progress = (epoch + 1 - start_epoch) / (epochs - start_epoch)
            filled = int(30 * progress)
            bar = 'x' * filled + '-' * (30 - filled)
            print(f"   Progress: [{bar}] {progress*100:.1f}% ({epoch+1}/{epochs})")
            
            # Save best model
            val_miou = val_results['mIou']
            if val_miou > self.best_val_miou:
                self.best_val_miou = val_miou
                self.best_epoch = epoch
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'best_val_miou': self.best_val_miou, 'config': self.config,
                    'train_history': self.train_history, 'val_history': self.val_history
                }
                if ema: checkpoint['ema'] = ema.state_dict()
                if scheduler: checkpoint['scheduler'] = scheduler.state_dict()
                
                torch.save(checkpoint, os.path.join(self.exp_dir, 'best_model.pth.tar'))
                print(f"   New best! mIoU: {self.best_val_miou*100:.2f}%")
            else:
                print(f"   Best: {self.best_val_miou*100:.2f}% (Epoch {self.best_epoch+1})")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save final results
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        history_data = {
            'train_history': self.train_history, 'val_history': self.val_history,
            'best_val_miou': self.best_val_miou, 'best_epoch': self.best_epoch
        }
        with open(os.path.join(self.exp_dir, 'training_history.json'), 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        print(f"\n Training completed!")
        print(f" Best mIoU: {self.best_val_miou*100:.2f}% (Epoch {self.best_epoch+1})")
        print(f" Saved to: {self.exp_dir}")
        
        return {
            'exp_name': self.exp_name, 'exp_dir': self.exp_dir,
            'best_val_miou': self.best_val_miou, 'best_epoch': self.best_epoch,
            'train_history': self.train_history, 'val_history': self.val_history
        }

def run_experiment(config):
    """Convenience function to run a training experiment"""
    return TrainingExperiment(config).run()
