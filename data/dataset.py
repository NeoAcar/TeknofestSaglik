import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pydicom
import numpy as np
import torchvision.transforms as transforms


class StrokeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, custom_transform=None):
        """
        image_paths: list of file paths (.png or .dcm)
        labels: list of binary labels (0 or 1)
        transform: preprocessing function (from BiomedCLIP)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.custom_transform = custom_transform

        self.combined_transform = None
        if self.transform and self.custom_transform:
            self.combined_transform = transforms.Compose([self.transform, self.custom_transform])
        elif self.transform:
            self.combined_transform = self.transform
        elif self.custom_transform:
            self.combined_transform = self.custom_transform
            
    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        if path.endswith(".dcm"):
            img = self._load_dicom(path)
        else:
            img = Image.open(path).convert("RGB")

        if self.combined_transform:
            img = self.combined_transform(img)
        elif self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)
        
    def _load_dicom(self, path):
        """
        Loads a DICOM file and returns a PIL Image (3-channel RGB).
        """
        dcm = pydicom.dcmread(path)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Normalize to [0, 255] and convert to uint8
        pixel_array -= pixel_array.min()
        pixel_array /= (pixel_array.max() + 1e-5)
        pixel_array *= 255.0
        pixel_array = pixel_array.astype(np.uint8)

        # Convert to RGB by stacking (ViT expects 3 channels)
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array]*3, axis=-1)

        return Image.fromarray(pixel_array)