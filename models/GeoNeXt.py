import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvNeXtV2 import ConvNeXtV2

class ConvNeXtV2Encoder(nn.Module):
    """ConvNeXtV2 encoder that extracts multi-scale features"""
    def __init__(self, in_channels=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1):
        super().__init__()
        self.backbone = ConvNeXtV2(
            in_chans=in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate
        )
        # Remove the final classification layers
        self.backbone.norm = nn.Identity()
        self.backbone.head = nn.Identity()
        
    def forward(self, x):
        """Extract features at multiple scales"""
        features = []
        
        # Stage 0: Stem (4x downsampling)
        x = self.backbone.downsample_layers[0](x)
        features.append(x)  # 1/4 resolution
        
        # Stage 1
        x = self.backbone.stages[0](x)
        features.append(x)  # 1/4 resolution
        
        # Stage 2 
        x = self.backbone.downsample_layers[1](x)
        x = self.backbone.stages[1](x)
        features.append(x)  # 1/8 resolution
        
        # Stage 3
        x = self.backbone.downsample_layers[2](x)
        x = self.backbone.stages[2](x)
        features.append(x)  # 1/16 resolution
        
        # Stage 4
        x = self.backbone.downsample_layers[3](x)
        x = self.backbone.stages[3](x)
        features.append(x)  # 1/32 resolution
        
        return features


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class SEWeightModule(nn.Module):
    """Squeeze-and-Excitation module"""
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Global pooling branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        # Global pooling
        global_feature = self.global_pool(x)
        global_feature = self.branch5_conv(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), mode='bilinear', align_corners=True)
        
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class PSAModule(nn.Module):
    """Pyramid Squeeze Attention Module"""
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        # Ensure planes//4 is divisible by group sizes
        quarter_planes = planes // 4
        # Adjust groups to be compatible with channel count
        safe_groups = [min(g, quarter_planes) for g in conv_groups]
        safe_groups = [g if quarter_planes % g == 0 else 1 for g in safe_groups]
        
        self.conv_1 = conv3x3(inplans, quarter_planes, stride=stride)
        self.conv_2 = nn.Conv2d(inplans, quarter_planes, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                               stride=stride, groups=safe_groups[1])
        self.conv_3 = nn.Conv2d(inplans, quarter_planes, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                               stride=stride, groups=safe_groups[2])
        self.conv_4 = nn.Conv2d(inplans, quarter_planes, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                               stride=stride, groups=safe_groups[3])
        self.se = SEWeightModule(quarter_planes)
        self.split_channel = quarter_planes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        out = feats_weight.reshape(batch_size, -1, feats_weight.shape[3], feats_weight.shape[4])

        return out
    

class PSABlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv1x1(in_ch, out_ch)
        self.bn1   = norm_layer(out_ch)

        self.psa = PSAModule(out_ch, out_ch)
        self.bn2  = norm_layer(out_ch)
        self.conv3 = conv1x1(out_ch, out_ch)
        self.bn3   = norm_layer(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                conv1x1(in_ch, out_ch, stride), norm_layer(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.psa(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class GeoNeXt(nn.Module):
    """Pyramid Squeeze Attention U-Net for semantic segmentation.
    
    Combines ConvNeXtV2 encoder with PSA attention blocks and ASPP
    for multi-scale feature processing.
    """
    def __init__(self, args):
        super().__init__()
        
        # Parse arguments
        depths = getattr(args, "convnext_depths", [3, 3, 9, 3])
        dims = getattr(args, "convnext_dims", [96, 192, 384, 768])
        drop_path_rate = getattr(args, "drop_path_rate", 0.1)
        n_channels = getattr(args, "n_channels", 3)
        freeze_encoder = getattr(args, "freeze_encoder", False)

        # ConvNeXtV2 encoder
        self.encoder = ConvNeXtV2Encoder(
            in_channels=n_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate
        )
        
        # Load pretrained weights if provided
        if hasattr(args, 'convnextv2_pretrained_weights') and args.convnextv2_pretrained_weights:
            self._load_pretrained_weights(args.convnextv2_pretrained_weights)
        
        # Apply encoder freezing based on configuration
        if freeze_encoder is not False:
            frozen_info = self._freeze_encoder_progressive(freeze_encoder)
        
        # PSA-based decoder
        self.upconv1 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
        self.psa_conv1 = PSABlock(dims[2]*2, dims[2])
        
        self.upconv2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
        self.psa_conv2 = PSABlock(dims[1]*2, dims[1])
        
        self.upconv3 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2)
        self.psa_conv3 = PSABlock(dims[0]*2, dims[0])
        
        # Final upsampling to compensate for stem's 4x downsampling
        self.final_upconv = nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4)
        
        # ASPP module for multi-scale context
        self.aspp = ASPP(dims[0], 64)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def _load_pretrained_weights(self, weights_path):
        """Load ConvNeXtV2 pretrained weights with GRN shape fixing"""
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out the head weights since we'll replace it
            backbone_state_dict = {k: v for k, v in state_dict.items() 
                                 if not k.startswith('head.')}
            
            # Fix GRN parameter shapes (convert from [1, dim] to [1, 1, 1, dim])
            def fix_grn_shapes(state_dict):
                fixed_dict = {}
                fixes_applied = 0
                for key, value in state_dict.items():
                    if '.grn.gamma' in key or '.grn.beta' in key:
                        if len(value.shape) == 2 and value.shape[0] == 1:
                            # Reshape from [1, dim] to [1, 1, 1, dim]
                            fixed_dict[key] = value.reshape(1, 1, 1, value.shape[1])
                            fixes_applied += 1
                        else:
                            fixed_dict[key] = value
                    else:
                        fixed_dict[key] = value
                return fixed_dict
            
            backbone_state_dict = fix_grn_shapes(backbone_state_dict)
            
            # Load weights with strict=False to allow missing head weights
            missing_keys, unexpected_keys = self.encoder.backbone.load_state_dict(backbone_state_dict, strict=False)
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    def _freeze_encoder_progressive(self, freeze_config):
        """
        Progressive encoder freezing with percentage-based, stage-based, or complete freezing.
        Freezes from deepest to shallowest layers to preserve most important feature extractions.
        
        Args:
            freeze_config: Can be:
                - True/False: Complete freezing or no freezing
                - float (0.0-1.0): Percentage of encoder to freeze from deepest layers
                - int (1-4): Number of stages to freeze from deepest layers  
                - dict: {"percentage": 0.5} or {"stages": [3, 4]} for explicit control
        
        Returns:
            dict: Information about what was frozen
        """
        if freeze_config is False:
            return {"frozen_stages": [], "frozen_params": 0, "total_params": self._count_total_encoder_parameters(), 
                   "message": "No freezing applied"}
        
        # Get encoder stage organization
        encoder_stages = self._get_encoder_stages()
        total_params = sum(stage['param_count'] for stage in encoder_stages)
        
        if freeze_config is True:
            # Freeze everything
            frozen_stages = list(range(len(encoder_stages)))
        elif isinstance(freeze_config, (int, float)):
            if isinstance(freeze_config, float):
                # Percentage-based freezing (0.0 to 1.0)
                if not 0.0 <= freeze_config <= 1.0:
                    freeze_config = 0.5
                
                # Calculate which stages to freeze to match the percentage
                frozen_stages = self._calculate_stages_for_percentage(encoder_stages, freeze_config)
            else:
                # Stage count (1-4)
                if not 1 <= freeze_config <= len(encoder_stages):
                    freeze_config = len(encoder_stages) // 2
                
                # Freeze deepest N stages
                frozen_stages = list(range(len(encoder_stages) - freeze_config, len(encoder_stages)))
        elif isinstance(freeze_config, dict):
            if 'percentage' in freeze_config:
                percentage = freeze_config['percentage']
                if not 0.0 <= percentage <= 1.0:
                    percentage = 0.5
                frozen_stages = self._calculate_stages_for_percentage(encoder_stages, percentage)
            elif 'stages' in freeze_config:
                # Explicit stage numbers (0-indexed)
                specified_stages = freeze_config['stages']
                frozen_stages = [s for s in specified_stages if 0 <= s < len(encoder_stages)]
            else:
                return {"frozen_stages": [], "frozen_params": 0, "total_params": total_params, 
                       "message": "Invalid config - no freezing applied"}
        else:
            return {"frozen_stages": [], "frozen_params": 0, "total_params": total_params, 
                   "message": "Invalid config type - no freezing applied"}
        
        # Apply freezing
        frozen_params = 0
        frozen_stage_names = []
        
        for stage_idx in frozen_stages:
            if 0 <= stage_idx < len(encoder_stages):
                stage_info = encoder_stages[stage_idx]
                # Freeze all parameters in this stage
                for param in stage_info['module'].parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                frozen_stage_names.append(stage_info['name'])
        
        # Create summary message
        if not frozen_stages:
            message = "No stages frozen"
        else:
            frozen_percentage = (frozen_params / total_params) * 100
            stage_names_str = ", ".join(frozen_stage_names)
            message = f"{len(frozen_stages)}/{len(encoder_stages)} stages frozen ({frozen_percentage:.1f}% of encoder): {stage_names_str}"
        
        return {
            "frozen_stages": frozen_stages,
            "frozen_stage_names": frozen_stage_names,
            "frozen_params": frozen_params,
            "total_params": total_params,
            "frozen_percentage": (frozen_params / total_params) * 100 if total_params > 0 else 0,
            "message": message
        }
    
    def _get_encoder_stages(self):
        """
        Get organized information about encoder stages for freezing control.
        Returns stages from shallowest to deepest (index 0 = shallowest, higher index = deeper).
        """
        stages = []
        
        # Stage 0: Stem (shallowest - first feature extraction)
        stem_module = self.encoder.backbone.downsample_layers[0]
        stem_params = sum(p.numel() for p in stem_module.parameters())
        stages.append({
            "name": "Stem",
            "module": stem_module,
            "param_count": stem_params,
            "description": "Initial 4x downsampling and feature extraction"
        })
        
        # Stage 1: ConvNeXt Stage 1  
        stage1_modules = [self.encoder.backbone.stages[0]]
        stage1_params = sum(p.numel() for module in stage1_modules for p in module.parameters())
        stages.append({
            "name": "Stage1", 
            "module": nn.ModuleList(stage1_modules),
            "param_count": stage1_params,
            "description": "First ConvNeXt stage (1/4 resolution)"
        })
        
        # Stage 2: ConvNeXt Stage 2
        stage2_modules = [self.encoder.backbone.downsample_layers[1], self.encoder.backbone.stages[1]]
        stage2_params = sum(p.numel() for module in stage2_modules for p in module.parameters())
        stages.append({
            "name": "Stage2",
            "module": nn.ModuleList(stage2_modules), 
            "param_count": stage2_params,
            "description": "Second ConvNeXt stage (1/8 resolution)"
        })
        
        # Stage 3: ConvNeXt Stage 3 (deeper features)
        stage3_modules = [self.encoder.backbone.downsample_layers[2], self.encoder.backbone.stages[2]]
        stage3_params = sum(p.numel() for module in stage3_modules for p in module.parameters())
        stages.append({
            "name": "Stage3",
            "module": nn.ModuleList(stage3_modules),
            "param_count": stage3_params,
            "description": "Third ConvNeXt stage (1/16 resolution) - deep features"
        })
        
        # Stage 4: ConvNeXt Stage 4 (deepest features - most important)
        stage4_modules = [self.encoder.backbone.downsample_layers[3], self.encoder.backbone.stages[3]]
        stage4_params = sum(p.numel() for module in stage4_modules for p in module.parameters())
        stages.append({
            "name": "Stage4",
            "module": nn.ModuleList(stage4_modules),
            "param_count": stage4_params,
            "description": "Fourth ConvNeXt stage (1/32 resolution) - deepest features"
        })
        
        return stages
    
    def _calculate_stages_for_percentage(self, encoder_stages, target_percentage):
        """
        Calculate which stages to freeze to achieve approximately the target percentage.
        Prioritizes freezing from deepest to shallowest stages.
        """
        total_params = sum(stage['param_count'] for stage in encoder_stages)
        target_params = total_params * target_percentage
        
        frozen_stages = []
        cumulative_params = 0
        
        for stage_idx in reversed(range(len(encoder_stages))):
            stage_params = encoder_stages[stage_idx]['param_count']
            if cumulative_params + stage_params <= target_params + (total_params * 0.05):  # 5% tolerance
                frozen_stages.append(stage_idx)
                cumulative_params += stage_params
            elif not frozen_stages: 
                frozen_stages.append(stage_idx)
                break
                
        return sorted(frozen_stages)

    def _freeze_encoder(self):
        """Legacy method - freeze entire encoder (kept for compatibility)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _unfreeze_encoder(self):
        """Unfreeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def _count_frozen_parameters(self):
        """Count the number of frozen parameters in the encoder"""
        return sum(param.numel() for param in self.encoder.parameters() if not param.requires_grad)
    
    def _count_total_encoder_parameters(self):
        """Count the total number of parameters in the encoder"""
        return sum(param.numel() for param in self.encoder.parameters())

    def forward(self, x):
        # Store original size for final resizing
        original_size = x.shape[2:]
        
        # Get multi-scale features from ConvNeXt encoder
        features = self.encoder(x)
        # features[0]: stem features (1/4 resolution)
        # features[1]: stage1 features (1/4 resolution)  
        # features[2]: stage2 features (1/8 resolution)
        # features[3]: stage3 features (1/16 resolution)
        # features[4]: stage4 features (1/32 resolution)
        
        # PSA-based decoder
        # Stage 4 -> Stage 3
        x = self.upconv1(features[4])  # Upsample to 1/16 resolution
        
        # Handle size mismatch
        if x.shape[2:] != features[3].shape[2:]:
            x = F.interpolate(x, size=features[3].shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, features[3]], dim=1)  # Skip connection
        x = self.psa_conv1(x)  # PSA attention processing
        
        # Stage 3 -> Stage 2
        x = self.upconv2(x)  # Upsample to 1/8 resolution
        
        # Handle size mismatch
        if x.shape[2:] != features[2].shape[2:]:
            x = F.interpolate(x, size=features[2].shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, features[2]], dim=1)  # Skip connection
        x = self.psa_conv2(x)  # PSA attention processing
        
        # Stage 2 -> Stage 1
        x = self.upconv3(x)  # Upsample to 1/4 resolution
        
        # Handle size mismatch
        if x.shape[2:] != features[1].shape[2:]:
            x = F.interpolate(x, size=features[1].shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, features[1]], dim=1)  # Skip connection
        x = self.psa_conv3(x)  # PSA attention processing
        
        # Final upsampling to original resolution
        x = self.final_upconv(x)
        
        # Handle any remaining size mismatch
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # ASPP for multi-scale context
        x = self.aspp(x)  # Multi-scale feature fusion
        x = self.final_conv(x)
        
        return x


# Convenience functions for different GeoNeXt variants
def geonext_tiny(args):
    """GeoNeXt with ConvNeXtV2-Tiny encoder"""
    args.convnext_depths = [3, 3, 9, 3]
    args.convnext_dims = [96, 192, 384, 768]
    return GeoNeXt(args)


def geonext_small(args):
    """GeoNeXt with ConvNeXtV2-Small encoder"""
    args.convnext_depths = [3, 3, 27, 3]
    args.convnext_dims = [96, 192, 384, 768]
    return GeoNeXt(args)


def geonext_base(args):
    """GeoNeXt with ConvNeXtV2-Base encoder"""
    args.convnext_depths = [3, 3, 27, 3]
    args.convnext_dims = [128, 256, 512, 1024]
    return GeoNeXt(args)


def geonext_large(args):
    """GeoNeXt with ConvNeXtV2-Large encoder"""
    args.convnext_depths = [3, 3, 27, 3]
    args.convnext_dims = [192, 384, 768, 1536]
    return GeoNeXt(args)