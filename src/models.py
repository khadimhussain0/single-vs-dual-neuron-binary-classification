"""
Neural network models for binary classification research.
This module implements various CNN architectures with both single-neuron
and two-neuron output layers for comparison using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class SmallCNN(nn.Module):
    """
    A small custom CNN for binary classification with configurable output layer.
    Useful for experiments that don't require heavy computation.
    """
    def __init__(self, input_channels=3, output_neurons=1):
        super(SmallCNN, self).__init__()
        
        if output_neurons not in [1, 2]:
            raise ValueError("output_neurons must be either 1 or 2")
            
        self.output_neurons = output_neurons
        
        self.conv1_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.4)
        
        # Calculate the flattened feature size for a 32x32 input
        # After 3 max-pooling layers of stride 2: 32 -> 16 -> 8 -> 4
        # With 128 channels: 128 * 4 * 4 = 2048
        self.feature_size = 128 * 4 * 4
        
        self.fc1 = nn.Linear(self.feature_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        
        if output_neurons == 1:
            self.fc_out = nn.Linear(256, 1)  # Single neuron
        else:
            self.fc_out = nn.Linear(256, 2)  # Two neurons
    
    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = x.view(-1, self.feature_size)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout4(x)
        
        x = self.fc_out(x)
        
        if self.output_neurons == 1:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
            
        return x


class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) model for binary classification with configurable output layer.
    """
    def __init__(self, input_channels=3, output_neurons=1, pretrained=True):
        super(ViTModel, self).__init__()
        
        if output_neurons not in [1, 2]:
            raise ValueError("output_neurons must be either 1 or 2")
            
        self.output_neurons = output_neurons
        
        if pretrained:
            self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.base_model = models.vit_b_16(weights=None)
            
        for name, param in self.base_model.named_parameters():
            if 'heads' not in name and 'encoder.layers.11' not in name:
                param.requires_grad = False
                
        hidden_size = self.base_model.heads.head.in_features
        
        self.base_model.heads = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        
        if output_neurons == 1:
            self.fc_out = nn.Linear(128, 1)  # Single neuron
        else:
            self.fc_out = nn.Linear(128, 2)  # Two neurons
            
    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        x = self.fc_out(x)
        
        # Apply activation based on output neurons
        if self.output_neurons == 1:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
            
        return x


class ResNetModel(nn.Module):
    """
    ResNet50-based model for binary classification with configurable output layer.
    """
    def __init__(self, input_channels=3, output_neurons=1, pretrained=True):
        super(ResNetModel, self).__init__()
        
        if output_neurons not in [1, 2]:
            raise ValueError("output_neurons must be either 1 or 2")
            
        self.output_neurons = output_neurons
        
        if pretrained:
            self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.base_model = models.resnet50(weights=None)
            
        for name, param in self.base_model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
                
        in_features = self.base_model.fc.in_features
        
        self.base_model.fc = nn.Identity()
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        # Output layer
        if output_neurons == 1:
            self.fc_out = nn.Linear(128, 1)  # Single neuron
        else:
            self.fc_out = nn.Linear(128, 2)  # Two neurons
            
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)
        x = self.fc_out(x)
        
        # Apply activation based on output neurons
        if self.output_neurons == 1:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
            
        return x


def create_model(model_name, input_channels=3, output_neurons=1, pretrained=True):
    """
    Factory function to create a model based on the specified architecture.
    
    Args:
        model_name: Name of the model architecture ('small_cnn', 'vit', 'resnet50')
        input_channels: Number of input channels in the images
        output_neurons: Number of neurons in output layer (1 or 2)
        pretrained: Whether to use pre-trained weights for transfer learning
        
    Returns:
        PyTorch model instance
    """
    if model_name.lower() == 'small_cnn':
        return SmallCNN(input_channels=input_channels, output_neurons=output_neurons)
    elif model_name.lower() == 'vit':
        return ViTModel(input_channels=input_channels, output_neurons=output_neurons, pretrained=pretrained)
    elif model_name.lower() == 'resnet50':
        return ResNetModel(input_channels=input_channels, output_neurons=output_neurons, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def get_loss_function(output_neurons):
    """
    Returns appropriate loss function based on the number of output neurons.
    
    Args:
        output_neurons: Number of output neurons (1 or 2)
        
    Returns:
        PyTorch loss function
    """
    if output_neurons == 1:
        # Single neuron: binary cross entropy
        return nn.BCELoss()
    else:
        # Two neurons: cross entropy
        return nn.CrossEntropyLoss()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_single = create_model('small_cnn', output_neurons=1)
    model_single.to(device)
    print(model_single)
    
    model_dual = create_model('small_cnn', output_neurons=2)
    model_dual.to(device)
    print(model_dual)
    
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    
    single_output = model_single(dummy_input)
    print(f"Single neuron output shape: {single_output.shape}")  
    
    dual_output = model_dual(dummy_input)
    print(f"Dual neuron output shape: {dual_output.shape}") 
