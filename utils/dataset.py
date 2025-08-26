
import os
import h5py
import numpy as np
import torch
import torchvision

class h5_Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_dir, train = True, augmentation_transform=None):
        if train:
            self.images_dir = h5_dir + os.sep + 'train'
            img_list    = sorted(os.listdir(self.images_dir))
            self.images = [i for i in img_list if i.endswith('.h5')]

        else:
            self.images_dir = h5_dir + os.sep + 'val'
            img_list    = sorted(os.listdir(self.images_dir))
            self.images = [i for i in img_list if i.endswith('.h5')]

        self.img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
        
        # Store augmentation transform (only applied during training)
        self.augmentation_transform = augmentation_transform if train else None
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        h5data  = h5py.File(self.images_dir + os.sep + self.images[i])
        rgb = h5data['image'][:]
        lab = h5data['label'][:]
        h5data.close()
        img = self.img_transform(rgb)

        lab = np.where(lab <= 128, 0, 1)
        lab = torch.from_numpy(lab)
        lab = torch.unsqueeze(lab, dim=0)
        
        # Apply augmentation transforms if available
        if self.augmentation_transform is not None:
            img, lab = self.augmentation_transform(img, lab)
        
        return img,  lab

class h5_CombinedDataset(torch.utils.data.Dataset):
    """Combined dataset that can load from multiple dataset directories"""
    def __init__(self, dataset_names, train=True, augmentation_transform=None):
        """
        Args:
            dataset_names: List of dataset names or single dataset name
            train: Whether to load training or validation data
            augmentation_transform: Augmentation transform to apply
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        self.dataset_names = dataset_names
        self.datasets = []
        self.dataset_lengths = []
        self.cumulative_lengths = []
        
        total_length = 0
        # Compute a base path relative to this file.  This avoids
        # hard‑coding any absolute paths and allows the repository to be
        # relocated without code changes.  The expected structure is
        # ``<repo>/dataset/<dataset_name>/{train,val}/*.h5``.
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
        for dataset_name in dataset_names:
            dataset_dir = os.path.join(base_dir, dataset_name)
            if os.path.isdir(dataset_dir):
                dataset = h5_Dataset(dataset_dir, train=train, augmentation_transform=augmentation_transform)
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(dataset))
                total_length += len(dataset)
                self.cumulative_lengths.append(total_length)
            else:
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if not self.datasets:
            raise ValueError(f"No valid datasets found from: {dataset_names}")
    
    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def __getitem__(self, idx):
        """Get item from the appropriate dataset based on index"""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumulative_length in enumerate(self.cumulative_lengths):
            if idx < cumulative_length:
                dataset_idx = i
                break
        
        # Calculate the local index within the selected dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        
        # Get item from the appropriate dataset
        return self.datasets[dataset_idx][local_idx]

# Matplotlib is intentionally not imported in this reduced version.
def display_images_with_predictions_and_labels(image1, prediction1, label1):
    """
    Placeholder for an interactive debugging aid to visualise images,
    predictions and labels side by side.  In this pared‑down version of
    the repository the implementation is intentionally omitted to avoid
    heavy dependencies such as Matplotlib.  Users who require visual
    inspection of results can implement this function using their
    preferred plotting library.
    """
    raise NotImplementedError(
        "display_images_with_predictions_and_labels is not implemented in the reduced codebase"
    )

def make_data_loaders(args):
    import os as _os
    # Build the dataset path relative to this file rather than assuming
    # the current working directory.  This mirrors the logic used in
    # ``h5_CombinedDataset`` above.
    data_path = _os.path.join(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', 'dataset')), args.dataset)

    Training_Data = h5_Dataset(data_path, True)
    valing_Data   = h5_Dataset(data_path, False)

    train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=args.batch_size, shuffle=False,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valing_Data, batch_size=args.batch_size, shuffle=False,drop_last=True)
    return train_loader, valid_loader
