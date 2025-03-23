import torchvision.transforms as transforms
import torch

def get_stroke_dataset_transforms():
    """
    Returns a dictionary of transformations for the StrokeDataset.
    Includes transformations for both training and validation/test datasets.
    """
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(5),  # Random rotation
        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.03),
        transforms.RandomAffine(0, shear=5, scale=(0.8, 1.2)),  # Random affine transformation
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Random perspective transformation
        # transforms.ToTensor(),  # Convert image to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    val_transforms = transforms.Compose([
        # transforms.Resize(256),  # Resize the shorter side to 256
        # transforms.CenterCrop(224),  # Center crop to 224x224
        # transforms.ToTensor(),  # Convert image to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    return {
        'train': train_transforms,
        'val': val_transforms
    }