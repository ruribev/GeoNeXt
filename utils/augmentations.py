import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image


class SegmentationTransform:
    """Base class for transforms that need to be applied to both image and mask"""
    
    def __call__(self, image, mask):
        raise NotImplementedError


class RandomHorizontalFlip(SegmentationTransform):
    """Random horizontal flip for both image and mask"""
    
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, image, mask):
        if random.random() < self.probability:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip(SegmentationTransform):
    """Random vertical flip for both image and mask"""
    
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, image, mask):
        if random.random() < self.probability:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class RandomRotation(SegmentationTransform):
    """Random rotation for both image and mask"""
    
    def __init__(self, degrees=15):
        self.degrees = degrees
    
    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        return image, mask


class ColorJitter(SegmentationTransform):
    """Color jittering only applied to image, mask unchanged"""
    
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast, 
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, image, mask):
        # Convert tensor to PIL for ColorJitter, then back to tensor
        if isinstance(image, torch.Tensor):
            image_pil = F.to_pil_image(image)
            image_jittered = self.color_jitter(image_pil)
            image = F.to_tensor(image_jittered)
        else:
            image = self.color_jitter(image)
        
        return image, mask


class ElasticTransform(SegmentationTransform):
    """Elastic transformation for both image and mask"""
    
    def __init__(self, alpha=120, sigma=6.0, probability=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.probability = probability
    
    def __call__(self, image, mask):
        if random.random() < self.probability:
            # Convert to PIL if tensor
            if isinstance(image, torch.Tensor):
                image_pil = F.to_pil_image(image)
                mask_pil = F.to_pil_image(mask.float())
            else:
                image_pil = image
                mask_pil = mask
            
            # Apply elastic transform - ensure alpha and sigma are floats
            elastic_transformer = transforms.ElasticTransform(
                alpha=float(self.alpha), 
                sigma=float(self.sigma),
                interpolation=F.InterpolationMode.BILINEAR
            )
            
            # Use same random state for both image and mask
            random_state = torch.get_rng_state()
            image_transformed = elastic_transformer(image_pil)
            
            torch.set_rng_state(random_state)
            mask_elastic_transformer = transforms.ElasticTransform(
                alpha=float(self.alpha),
                sigma=float(self.sigma), 
                interpolation=F.InterpolationMode.NEAREST
            )
            mask_transformed = mask_elastic_transformer(mask_pil)
            
            # Convert back to tensor
            if isinstance(image, torch.Tensor):
                image = F.to_tensor(image_transformed)
                mask = F.to_tensor(mask_transformed)
                # Ensure mask is binary and preserve integer dtype
                mask = (mask > 0.5).long()
            else:
                image = image_transformed
                mask = mask_transformed
        
        return image, mask


class GaussianBlur(SegmentationTransform):
    """Gaussian blur only applied to image, mask unchanged"""
    
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), probability=0.2):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.probability = probability
    
    def __call__(self, image, mask):
        if random.random() < self.probability:
            sigma_val = random.uniform(self.sigma[0], self.sigma[1])
            image = F.gaussian_blur(image, kernel_size=self.kernel_size, sigma=sigma_val)
        
        return image, mask


class RandomGamma(SegmentationTransform):
    """Random gamma adjustment only applied to image, mask unchanged"""
    
    def __init__(self, gamma_range=(0.8, 1.2), probability=0.25):
        self.gamma_range = gamma_range
        self.probability = probability
    
    def __call__(self, image, mask):
        if random.random() < self.probability:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            image = F.adjust_gamma(image, gamma)
        
        return image, mask


class GaussianNoise(SegmentationTransform):
    """Add Gaussian noise only to image, mask unchanged"""
    
    def __init__(self, std_range=(0.01, 0.03), probability=0.1):
        self.std_range = std_range
        self.probability = probability
    
    def __call__(self, image, mask):
        if random.random() < self.probability:
            std = random.uniform(self.std_range[0], self.std_range[1])
            if isinstance(image, torch.Tensor):
                noise = torch.randn_like(image) * std
                image = torch.clamp(image + noise, 0, 1)
            else:
                # For PIL image, convert to tensor, add noise, convert back
                image_tensor = F.to_tensor(image)
                noise = torch.randn_like(image_tensor) * std
                image_tensor = torch.clamp(image_tensor + noise, 0, 1)
                image = F.to_pil_image(image_tensor)
        
        return image, mask


class RandomErasing(SegmentationTransform):
    """Random erasing applied to both image and mask"""
    
    def __init__(self, scale_range=(0.02, 0.08), ratio_range=(0.3, 3.3), probability=0.15):
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.probability = probability
    
    def __call__(self, image, mask):
        if random.random() < self.probability:
            if isinstance(image, torch.Tensor):
                _, h, w = image.shape
            else:
                w, h = image.size
            
            # Calculate area to erase
            area = h * w
            target_area = random.uniform(self.scale_range[0], self.scale_range[1]) * area
            aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
            
            # Calculate dimensions
            erase_h = int(round((target_area * aspect_ratio) ** 0.5))
            erase_w = int(round((target_area / aspect_ratio) ** 0.5))
            
            if erase_h < h and erase_w < w:
                # Random position
                top = random.randint(0, h - erase_h)
                left = random.randint(0, w - erase_w)
                
                if isinstance(image, torch.Tensor):
                    # Fill with random values for image, zeros for mask
                    image[:, top:top+erase_h, left:left+erase_w] = torch.rand(3, erase_h, erase_w)
                    mask[:, top:top+erase_h, left:left+erase_w] = 0
                else:
                    # For PIL images, convert to tensor, apply erasing, convert back
                    image_tensor = F.to_tensor(image)
                    mask_tensor = F.to_tensor(mask.convert('L'))
                    
                    image_tensor[:, top:top+erase_h, left:left+erase_w] = torch.rand(3, erase_h, erase_w)
                    mask_tensor[:, top:top+erase_h, left:left+erase_w] = 0
                    
                    image = F.to_pil_image(image_tensor)
                    mask = F.to_pil_image(mask_tensor.squeeze())
        
        return image, mask


class RandomCropScale(SegmentationTransform):
    """Random crop and scale for both image and mask"""
    
    def __init__(self, scale_range=(0.8, 1.0), probability=0.4):
        self.scale_range = scale_range
        self.probability = probability
    
    def __call__(self, image, mask):
        if random.random() < self.probability:
            # Get original size
            if isinstance(image, torch.Tensor):
                _, h, w = image.shape
            else:
                w, h = image.size
            
            # Random scale
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Random crop position
            if new_h < h and new_w < w:
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                
                image = F.crop(image, top, left, new_h, new_w)
                mask = F.crop(mask, top, left, new_h, new_w)
                
                # Resize back to original size
                image = F.resize(image, (h, w), interpolation=F.InterpolationMode.BILINEAR)
                mask = F.resize(mask, (h, w), interpolation=F.InterpolationMode.NEAREST)
        
        return image, mask


class SegmentationCompose:
    """Compose multiple segmentation transforms"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


def create_augmentation_transforms(augmentation_config):
    """Create augmentation transforms from configuration dictionary
    
    Args:
        augmentation_config (dict): Configuration dictionary with augmentation parameters
        
    Returns:
        SegmentationCompose: Composed transforms for training
    """
    transforms_list = []
    
    if 'horizontal_flip' in augmentation_config:
        transforms_list.append(
            RandomHorizontalFlip(
                probability=augmentation_config['horizontal_flip'].get('probability', 0.5)
            )
        )
    
    if 'vertical_flip' in augmentation_config:
        transforms_list.append(
            RandomVerticalFlip(
                probability=augmentation_config['vertical_flip'].get('probability', 0.5)
            )
        )
    
    if 'rotation' in augmentation_config:
        transforms_list.append(
            RandomRotation(
                degrees=augmentation_config['rotation'].get('degrees', 15)
            )
        )
    
    if 'random_rotation' in augmentation_config:
        transforms_list.append(
            RandomRotation(
                degrees=augmentation_config['random_rotation'].get('degrees', 15)
            )
        )
    
    if 'color_jitter' in augmentation_config:
        jitter_params = augmentation_config['color_jitter']
        transforms_list.append(
            ColorJitter(
                brightness=jitter_params.get('brightness', 0.3),
                contrast=jitter_params.get('contrast', 0.3),
                saturation=jitter_params.get('saturation', 0.1),
                hue=jitter_params.get('hue', 0.05)
            )
        )
    
    if 'random_gamma' in augmentation_config:
        gamma_params = augmentation_config['random_gamma']
        transforms_list.append(
            RandomGamma(
                gamma_range=gamma_params.get('gamma_range', [0.8, 1.2]),
                probability=gamma_params.get('probability', 0.25)
            )
        )
    
    if 'gaussian_noise' in augmentation_config:
        noise_params = augmentation_config['gaussian_noise']
        transforms_list.append(
            GaussianNoise(
                std_range=noise_params.get('std_range', [0.01, 0.03]),
                probability=noise_params.get('probability', 0.1)
            )
        )
    
    if 'random_erasing' in augmentation_config:
        erasing_params = augmentation_config['random_erasing']
        transforms_list.append(
            RandomErasing(
                scale_range=erasing_params.get('scale_range', [0.02, 0.08]),
                ratio_range=erasing_params.get('ratio_range', [0.3, 3.3]),
                probability=erasing_params.get('probability', 0.15)
            )
        )
    
    if 'elastic_transform' in augmentation_config:
        elastic_params = augmentation_config['elastic_transform']
        transforms_list.append(
            ElasticTransform(
                alpha=elastic_params.get('alpha', 120),
                sigma=elastic_params.get('sigma', 6.0),
                probability=elastic_params.get('probability', 0.3)
            )
        )
    
    if 'gaussian_blur' in augmentation_config:
        blur_params = augmentation_config['gaussian_blur']
        # Handle both list and tuple formats for sigma
        sigma = blur_params.get('sigma', [0.1, 2.0])
        if isinstance(sigma, list) and len(sigma) == 2:
            sigma = tuple(sigma)
        elif not isinstance(sigma, tuple):
            sigma = (0.1, 2.0)
        
        # Handle kernel_size_range parameter
        kernel_size = blur_params.get('kernel_size', 3)
        if 'kernel_size_range' in blur_params:
            kernel_range = blur_params['kernel_size_range']
            if isinstance(kernel_range, list) and len(kernel_range) == 2:
                kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))  # Ensure odd
            else:
                kernel_size = kernel_range
        
        transforms_list.append(
            GaussianBlur(
                kernel_size=kernel_size,
                sigma=sigma,
                probability=blur_params.get('probability', 0.2)
            )
        )
    
    if 'random_crop_scale' in augmentation_config:
        crop_params = augmentation_config['random_crop_scale']
        transforms_list.append(
            RandomCropScale(
                scale_range=crop_params.get('scale_range', [0.8, 1.0]),
                probability=crop_params.get('probability', 0.4)
            )
        )
    
    return SegmentationCompose(transforms_list) if transforms_list else None
