import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from typing import Optional, Dict, Any

from models import geonext_tiny
from utils import h5_Dataset, evaluator
from utils.ema import ModelEMA

class Benchmark:
    """
    Simple benchmark class for GeoNeXt models
    """
    
    def __init__(self, dataset: str, gpu_id: int, model_variant: str, model_weight_path: str,
                 batch_size: int = 8, num_classes: int = 2, 
                 data_path: Optional[str] = None):
        """
        Initialize benchmark with configuration
        
        Args:
            dataset: Name of the dataset
            gpu_id: GPU ID to use (-1 for CPU)
            model_variant: Model variant ('tiny')
            model_weight_path: Path to model weights file
            batch_size: Batch size for evaluation
            num_classes: Number of classes in the dataset
            data_path: Optional custom data path
        """
        self.dataset = dataset
        self.gpu_id = gpu_id
        self.model_variant = model_variant
        self.model_weight_path = model_weight_path
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.data_path = data_path

        # Initialize attributes
        self.model = None
        self.device = None
        self.val_loader = None
        self.evaluator_obj = None
        self.results = None
        self.ema = None

        # Setup
        self._setup_device()
        self._setup_model()
        self._setup_data()
        self.evaluator_obj = evaluator(self.num_classes)

        print(f"Benchmark initialized for {self.dataset} dataset")
        print(f"   Model: {self.model_variant}")
        print(f"   Device: {self.device}")
        print(f"   Weights: {os.path.basename(self.model_weight_path)}")

    def _setup_device(self):
        """Setup computation device (GPU/CPU)"""
        if self.gpu_id == -1:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available() and self.gpu_id >= 0:
            self.device = torch.device(f'cuda:{self.gpu_id}')
        else:
            print(f"WARNING: GPU {self.gpu_id} not available, falling back to CPU")
            self.device = torch.device('cpu')
    
    def _setup_model(self):
        """Initialize and load the model"""
        # Create args for model initialization
        args = argparse.Namespace()
        args.dataset = self.dataset
        args.device = self.device
        args.weight_path = self.model_weight_path
        args.gpu_id = self.gpu_id

        # Create model based on variant
        if self.model_variant == 'tiny':
            print(f"Creating GeoNeXt-{self.model_variant} model...")
            self.model = geonext_tiny(args)
        else:
            raise ValueError(f"Unknown model variant: {self.model_variant}")
        
        self.model = self.model.to(self.device)
        
        # Load weights
        if os.path.exists(self.model_weight_path):
            try:
                # Add safe globals for numpy compatibility
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                checkpoint = torch.load(self.model_weight_path, map_location=self.device, weights_only=True)
            except Exception as e:
                print(f"WARNING: Safe loading failed: {e}")
                print("Falling back to weights_only=False (ensure model file is from trusted source)")
                checkpoint = torch.load(self.model_weight_path, map_location=self.device, weights_only=False)

            # Extract state dict (supports raw state_dict or full checkpoint)
            weight_state = checkpoint.get('state_dict', checkpoint)

            try:
                self.model.load_state_dict(weight_state)
                print(f"Successfully loaded model weights from {self.model_weight_path}")
            except RuntimeError as e:
                print(f"WARNING: Model loading failed with error: {e}")
                print("Attempting to load with strict=False...")
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(weight_state, strict=False)
                    if missing_keys:
                        print(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {unexpected_keys}")
                    print("Loaded with relaxed constraints")
                except Exception as e2:
                    print(f"ERROR: Failed to load model weights: {e2}")
                    raise e2

            # If EMA is present in checkpoint, set up EMA for equivalent results
            if isinstance(checkpoint, dict) and 'ema' in checkpoint and checkpoint['ema'] is not None:
                try:
                    self.ema = ModelEMA(self.model, device=self.device)
                    self.ema.load_state_dict(checkpoint['ema'])
                    print("Loaded EMA weights from checkpoint (will be used during evaluation)")
                except Exception as ema_e:
                    print(f"WARNING: Failed to load EMA from checkpoint: {ema_e}")

            self.model.eval()
        else:
            raise FileNotFoundError(f"Model weights not found: {self.model_weight_path}")
    
    def _setup_data(self):
        """Setup data loaders"""
        # Determine data path
        if not self.data_path:
            self.data_path = os.path.join(os.getcwd(), 'dataset', self.dataset)

        # Create validation dataset
        val_dataset = h5_Dataset(self.data_path, train=False, augmentation_transform=None)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )
        
        print(f'Loaded {len(val_dataset)} validation samples from {self.data_path}')
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run benchmark evaluation
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nRunning benchmark for {self.dataset} dataset...")

        # Reset evaluator
        self.evaluator_obj = evaluator(self.num_classes)

        # Use EMA weights if available
        use_ema = hasattr(self, 'ema') and self.ema is not None
        if use_ema:
            print("Using EMA weights for evaluation...")
            self.ema.apply_shadow()

        try:
            # Run evaluation
            start_time = time.time()
            with torch.no_grad():
                for imgs, labels in self.val_loader:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(imgs.float())
                    preds = torch.where(outputs > 0.5, 1, 0)
                    self.evaluator_obj.add_batch(labels, preds)
            
            end_time = time.time()
        finally:
            # Restore original weights
            if use_ema:
                self.ema.restore()
                print("Restored original weights after evaluation")

        # Collect metrics
        self.results = {
            'accuracy': self.evaluator_obj.OverallAccuracy(),
            'mIou': self.evaluator_obj.MeanIntersectionOverUnion(),
            'IoU': self.evaluator_obj.IntersectionOverUnion(),
            'precision': self.evaluator_obj.Precision(),
            'recall': self.evaluator_obj.Recall(),
            'f1': self.evaluator_obj.F1Score(),
            'duration': end_time - start_time,
            'model_variant': self.model_variant,
            'dataset': self.dataset,
            'device': str(self.device),
            'batch_size': self.batch_size,
        }

        return self.results
    
    def show_metrics(self) -> None:
        """Display benchmark metrics"""
        if self.results is None:
            print("ERROR: No benchmark results available. Run benchmark first.")
            return
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS - {self.dataset} Dataset")
        print(f"{'='*60}")
        print(f"Model Variant    : {self.results['model_variant']}")
        print(f"Device          : {self.results['device']}")
        print(f"Batch Size      : {self.results['batch_size']}")
        print(f"{'='*60}")
        
        avg_precision = (self.results['precision'][0] + self.results['precision'][1]) / 2.0
        avg_recall = (self.results['recall'][0] + self.results['recall'][1]) / 2.0
        avg_f1 = (float(self.results['f1'][0]) + float(self.results['f1'][1])) / 2.0
        
        print(f"PERFORMANCE METRICS:")
        print(f"   Overall Accuracy : {self.results['accuracy']*100:.2f}%")
        print(f"   Precision       : {avg_precision*100:.2f}%")
        print(f"   Recall          : {avg_recall*100:.2f}%")
        print(f"   F1 Score        : {avg_f1*100:.2f}%")
        print(f"   mIoU            : {float(self.results['mIou'])*100:.2f}%")
        print(f"   IoU (Class 1)   : {self.results['IoU'][1]*100:.2f}%")
        
        print(f"\nCLASS-SPECIFIC METRICS:")
        print(f"   Precision (Class 1): {self.results['precision'][1]*100:.2f}%")
        print(f"   Recall (Class 1)   : {self.results['recall'][1]*100:.2f}%")
        print(f"   F1 Score (Class 1) : {float(self.results['f1'][1])*100:.2f}%")
        
        print(f"\nTIMING INFORMATION:")
        print(f"   Validation Time : {self.results['duration']:.2f} seconds")
        
        print(f"\nDETAILED METRICS:")
        print(f"   Per-class Precision: {[f'{p:.4f}' for p in self.results['precision']]}")
        print(f"   Per-class Recall   : {[f'{r:.4f}' for r in self.results['recall']]}")
        print(f"   Per-class F1       : {[f'{f:.4f}' for f in self.results['f1']]}")
        print(f"   Per-class IoU      : {[f'{i:.4f}' for i in self.results['IoU']]}")
        
        print(f"{'='*60}")
    
    def show_examples(self, num_examples: int = 5, min_mask_percentage: float = 0.0) -> None:
        """
        Display visual examples of model predictions
        
        Args:
            num_examples: Number of examples to show
            min_mask_percentage: Minimum mask percentage to filter examples
        """
        print(f"\nGenerating {num_examples} visual examples...")
        if min_mask_percentage > 0.0:
            print(f"Filtering examples with mask percentage >= {min_mask_percentage:.1%}")
        
        val_loader_iter = iter(self.val_loader)
        examples_shown = 0
        batch_idx = 0
        examples_evaluated = 0
        
        while examples_shown < num_examples:
            try:
                x, y = next(val_loader_iter)
                batch_idx += 1
            except StopIteration:
                print(f"WARNING: Reached end of dataset. Showing {examples_shown} examples.")
                break
            
            rgb = x.to(self.device)
            
            with torch.no_grad():
                output = self.model(rgb.float())
            
            pred = torch.where(output > 0.5, 1, 0).cpu().numpy()
            
            batch_size = x.shape[0]
            for i in range(batch_size):
                if examples_shown >= num_examples:
                    break
                    
                # Calculate mask percentage
                mask = y[i].detach().numpy()
                total_pixels = mask.size
                positive_pixels = np.sum(mask > 0.5)
                mask_percentage = positive_pixels / total_pixels
                
                examples_evaluated += 1
                
                if mask_percentage >= min_mask_percentage:
                    print(f"\nExample {examples_shown + 1} (Batch {batch_idx}, Index {i}):")
                    print(f"   Mask percentage: {mask_percentage:.2%}")
                    
                    global_index = (batch_idx - 1) * batch_size + i
                    print(f"   Dataset index: {global_index}")
                    
                    self._display_example(x[i], pred[i], y[i], global_index)
                    examples_shown += 1
        
        print(f"\nSummary: Evaluated {examples_evaluated} examples, showed {examples_shown} examples")

    def _display_example(self, image, prediction, label, dataset_index: int):
        """Display a single example with image, prediction, and ground truth"""
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Image panel
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image
        
        if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
            image_np = np.transpose(image_np, (1, 2, 0))
        
        axs[0].imshow(image_np)
        axs[0].axis('off')
        axs[0].set_title(f'Image (Index: {dataset_index})')

        # Prediction panel
        if isinstance(prediction, torch.Tensor):
            pred_np = prediction.detach().cpu().numpy()
        else:
            pred_np = prediction
        
        axs[1].imshow(pred_np[0], cmap='gray')
        axs[1].axis('off')
        axs[1].set_title('Prediction')

        # Label panel
        if isinstance(label, torch.Tensor):
            lab_np = label.detach().cpu().numpy()
        else:
            lab_np = label
        
        axs[2].imshow(lab_np[0], cmap='gray')
        axs[2].axis('off')
        axs[2].set_title('Ground Truth')

        plt.tight_layout()
        plt.show()
