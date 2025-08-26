import torch
import torch.nn as nn
from typing import Dict, Optional


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    This implementation maintains a shadow copy of the model parameters
    and updates them using exponential moving average during training.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        decay: float = 0.9999, 
        device: Optional[torch.device] = None,
        update_freq: int = 1
    ):
        """
        Initialize ModelEMA.
        
        Args:
            model: The model to apply EMA to
            decay: EMA decay factor (higher = more smoothing)
            device: Device to store EMA parameters on
            update_freq: Update EMA every N steps (default: 1, update every step)
        """
        self.model = model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.update_freq = update_freq
        self.step_count = 0
        
        # Create shadow parameters
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters with model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    def update(self, model: Optional[nn.Module] = None) -> None:
        """
        Update EMA parameters.
        
        Args:
            model: Model to update from (if different from initialization)
        """
        self.step_count += 1
        
        # Only update every update_freq steps
        if self.step_count % self.update_freq != 0:
            return
            
        if model is None:
            model = self.model
            
        # Calculate actual decay (increases over time to stabilize training)
        decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA update: shadow = decay * shadow + (1 - decay) * current
                    self.shadow[name].mul_(decay).add_(param.data.to(self.device), alpha=1 - decay)
    
    def apply_shadow(self) -> None:
        """Apply EMA parameters to the model (for evaluation/inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self) -> None:
        """Restore original model parameters (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> Dict:
        """Get EMA state dictionary."""
        return {
            'decay': self.decay,
            'update_freq': self.update_freq,
            'step_count': self.step_count,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load EMA state dictionary."""
        self.decay = state_dict['decay']
        self.update_freq = state_dict['update_freq']
        self.step_count = state_dict['step_count']
        self.shadow = state_dict['shadow']
        
        # Move shadow parameters to the correct device
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(self.device)
    
    def to(self, device: torch.device) -> 'ModelEMA':
        """Move EMA to device."""
        self.device = device
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        return self
    
    def __enter__(self):
        """Context manager entry - apply shadow parameters."""
        self.apply_shadow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore original parameters."""
        self.restore()


def create_ema(
    model: nn.Module, 
    decay: float = 0.9999, 
    device: Optional[torch.device] = None,
    update_freq: int = 1
) -> ModelEMA:
    """
    Factory function to create ModelEMA instance.
    
    Args:
        model: The model to apply EMA to
        decay: EMA decay factor (default: 0.9999)
        device: Device to store EMA parameters on
        update_freq: Update EMA every N steps
        
    Returns:
        ModelEMA instance
    """
    return ModelEMA(model, decay, device, update_freq)
