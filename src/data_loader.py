"""
Dataset preparation for binary classification research.
This script prepares binary classification datasets for our experiments using PyTorch.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image

# Define CIFAR-10 class names for reference
CIFAR10_CLASSES = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

class BinaryClassificationDataset(Dataset):
    """
    PyTorch dataset for binary classification from a subset of CIFAR-10.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
            
        return image, label

def load_cifar10_binary(class_a=0, class_b=1, val_split=0.1, img_size=(224, 224)):
    """
    Load CIFAR-10 dataset for binary classification between two specified classes.
    
    Args:
        class_a: Index of the first class (default: 0, airplane)
        class_b: Index of the second class (default: 1, automobile)
        val_split: Validation split ratio from training data
        img_size: Target image size
        
    Returns:
        train_dataset, val_dataset, test_dataset: PyTorch Dataset objects
    """
    print(f"Loading CIFAR-10 binary classification: {CIFAR10_CLASSES[class_a]} vs {CIFAR10_CLASSES[class_b]}")
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    cifar_train = datasets.CIFAR10(root='../data', train=True, download=True, transform=None)
    cifar_test = datasets.CIFAR10(root='../data', train=False, download=True, transform=None)
    
    x_train = np.array([np.array(img) for img, _ in cifar_train]) / 255.0
    y_train = np.array([label for _, label in cifar_train])
    
    x_test = np.array([np.array(img) for img, _ in cifar_test]) / 255.0
    y_test = np.array([label for _, label in cifar_test])
    
    train_mask = np.logical_or(y_train == class_a, y_train == class_b)
    test_mask = np.logical_or(y_test == class_a, y_test == class_b)
    
    x_train_binary = x_train[train_mask]
    y_train_binary = y_train[train_mask]
    x_test_binary = x_test[test_mask]
    y_test_binary = y_test[test_mask]
    
    y_train_binary = (y_train_binary == class_b).astype(int)
    y_test_binary = (y_test_binary == class_b).astype(int)
    
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train_binary, y_train_binary, test_size=val_split, stratify=y_train_binary, random_state=42
    )
    
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    
    train_dataset = BinaryClassificationDataset(x_train_final, y_train_final, transform=train_transform)
    val_dataset = BinaryClassificationDataset(x_val, y_val, transform=val_test_transform)
    test_dataset = BinaryClassificationDataset(x_test_binary, y_test_binary, transform=val_test_transform)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Class distribution - Train: {np.bincount(y_train_final)}")
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=2):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: PyTorch Dataset for training
        val_dataset: PyTorch Dataset for validation
        test_dataset: PyTorch Dataset for testing
        batch_size: Batch size for training
        num_workers: Number of worker threads for data loading
        
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoader objects
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_cifar10_binary(class_a=0, class_b=1)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset
    )
    
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    print(f"Label distribution in batch: {torch.bincount(labels.long())}")
