"""CNN model definition
This module contains the implementation of the configurable CNN architecture
used for the iNaturalist classification task.
"""

import math
import torch
import torch.nn as nn

class CNN(nn.Module):
    """Configurable CNN architecture
    The model structure can be adjusted through the configuration dictionary,
    allowing for experimentation with different architectures.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv_blocks = nn.ModuleList()  # Stores convolutional blocks
        
        in_channels = 3  # Input channels (RGB images)
        current_size = cfg["img_size"]  # Track spatial dimensions through layers
        
        # Build convolutional blocks dynamically based on config
        for out_channels, kernel_size in zip(cfg["num_filters"], cfg["filter_sizes"]):
            block = self._create_conv_block(
                in_channels, out_channels, kernel_size,
                use_batchnorm=cfg["batch_norm"],
                activation=cfg["activation"]
            )
            self.conv_blocks.append(block)
            current_size = self._calculate_output(current_size, kernel_size)
            in_channels = out_channels
            
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))  # Fixed size output
        self.classifier = self._create_classifier(
            in_channels * 6 * 6,  # Flattened size after adaptive pooling
            cfg["fc_hidden_sizes"],
            10,  # Number of classes in iNaturalist-12K
            cfg["dropout"],
            cfg["batch_norm"],
            cfg["activation"]
        )
        
    def _calculate_output(self, input_size, kernel_size):
        """Calculate output dimensions after conv and pooling
        Args:
            input_size: Spatial dimension of input
            kernel_size: Size of convolutional kernel
        Returns:
            Output spatial dimension after convolution and max pooling
        """
        conv_size = math.floor((input_size - kernel_size + 2) / 1) + 1
        return math.floor((conv_size - 2) / 2) + 1
    
    def _create_conv_block(self, in_channels, out_channels, kernel_size, 
                          use_batchnorm=True, activation='relu'):
        """Create a convolutional block
        Each block consists of:
        - Conv2d layer
        - Optional batch normalization
        - Activation function
        - Max pooling
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            self._get_activation(activation),
            nn.MaxPool2d(2, 2)
        ]
        return nn.Sequential(*layers)
    
    def _create_classifier(self, in_features, hidden_size, num_classes,
                          dropout_rate, use_batchnorm, activation):
        """Create the classifier head
        Consists of:
        - Linear layer
        - Optional batch normalization
        - Activation function
        - Dropout
        - Final linear layer
        """
        layers = [
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size) if use_batchnorm else nn.Identity(),
            self._get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        ]
        return nn.Sequential(*layers)
    
    def _get_activation(self, name):
        """Get activation function by name
        Args:
            name: Name of activation function
        Returns:
            Corresponding activation function module
        """
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        return activations.get(name.lower(), nn.ReLU())  # Default to ReLU
    
    def forward(self, x):
        """Forward pass through the network"""
        for block in self.conv_blocks:
            x = block(x)
        x = self.adaptive_pool(x)  # Adaptive pooling to fixed size
        x = torch.flatten(x, 1)  # Flatten for classifier
        return self.classifier(x)

def initialize_model(cfg, device):
    """Create and configure model
    Args:
        cfg: Configuration dictionary
        device: Device to place model on
    Returns:
        Configured model, potentially wrapped in DataParallel
    """
    model = CNN(cfg)
    if torch.cuda.device_count() > 1:  # Use multiple GPUs if available
        model = nn.DataParallel(model)
    return model.to(device)