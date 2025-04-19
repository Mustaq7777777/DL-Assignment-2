"""Configuration settings and argument parsing
This module handles all configuration parameters for the training pipeline,
including command line argument parsing and default value management.
"""

import argparse
from pathlib import Path

# Default configuration values used when no command line arguments are provided
DEFAULT_CONFIG = {
    "batch_size": 32,
    "optimizer": "nadam",
    "learning_rate": 1e-4,
    "num_filters": [32, 64, 128, 256, 512],  # Number of filters for each conv layer
    "filter_sizes": [5, 5, 5, 5, 5],  # Kernel sizes for each conv layer
    "activation": "silu",  # Default activation function
    "fc_hidden_sizes": 512,  # Size of fully connected layer
    "dropout": 0.4,  # Dropout probability
    "batch_norm": True,  # Whether to use batch normalization
    "augmentation": False,  # Whether to use data augmentation
    "img_size": 400,  # Input image size
    "weight_decay": 0,  # L2 regularization strength
    "num_epochs": 8,  # Number of training epochs
    "base_dir": "/kaggle/input/nature-12k/inaturalist_12K"  # Dataset directory
}

def get_config():
    """Parse command-line arguments and return configuration
    Returns:
        tuple: (config dictionary, W&B entity, W&B project name)
    """
    parser = argparse.ArgumentParser(description="Train CNN on iNaturalist-12K")
    
    # WandB arguments
    parser.add_argument("--wandb_entity", "-we", 
                       default="cs24m045-indian-institute-of-technology-madras",
                       help="W&B Entity for logging")
    parser.add_argument("--wandb_project", "-wp", 
                       default="DA6401-Assignment-2",
                       help="W&B Project name")
    
    # Model parameters
    parser.add_argument("--num_epochs", "-e", type=int, 
                       default=DEFAULT_CONFIG["num_epochs"],
                       help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int,
                       default=DEFAULT_CONFIG["batch_size"],
                       help="Batch size")
    parser.add_argument("--augmentation", "-au", 
                       choices=["true","false"],
                       default=str(DEFAULT_CONFIG["augmentation"]).lower(),
                       help="Enable data augmentation")
    parser.add_argument("--learning_rate", "-lr", type=float,
                       default=DEFAULT_CONFIG["learning_rate"],
                       help="Learning rate")
    parser.add_argument("--weight_decay", "-w_d", type=float,
                       default=DEFAULT_CONFIG["weight_decay"],
                       help="L2 weight decay")
    parser.add_argument("--base_dir", "-br", type=str,
                       default=DEFAULT_CONFIG["base_dir"],
                       help="Dataset base directory")
    parser.add_argument("--activation", "-ac", 
                       choices=['relu','gelu','silu','mish','elu','selu'],
                       default=DEFAULT_CONFIG["activation"],
                       help="Activation function")
    parser.add_argument("--num_filters", "-nf", nargs=5, type=int,
                       default=DEFAULT_CONFIG["num_filters"],
                       help="List of filters per conv layer")
    parser.add_argument("--filter_sizes", "-fs", nargs=5, type=int,
                       default=DEFAULT_CONFIG["filter_sizes"],
                       help="List of kernel sizes per conv layer")
    parser.add_argument("--input_dim", "-in", type=int,
                       default=DEFAULT_CONFIG["img_size"],
                       help="Input image dimension")
    parser.add_argument("--batch_norm", "-bn", 
                       choices=["true","false"],
                       default=str(DEFAULT_CONFIG["batch_norm"]).lower(),
                       help="Enable batch normalization")
    parser.add_argument("--optimizer_name", "-o", 
                       choices=['nadam','adam','rmsprop'],
                       default=DEFAULT_CONFIG["optimizer"],
                       help="Optimizer choice")
    parser.add_argument("--hidden_size", "-dl", type=int,
                       default=DEFAULT_CONFIG["fc_hidden_sizes"],
                       help="Hidden units in classifier")
    parser.add_argument("--dropout", "-dp", type=float,
                       default=DEFAULT_CONFIG["dropout"],
                       help="Dropout probability")

    args = parser.parse_args()
    
    # Merge defaults with command-line arguments
    final_config = DEFAULT_CONFIG.copy()
    final_config.update({
        "batch_size": args.batch_size,
        "optimizer": args.optimizer_name,
        "learning_rate": args.learning_rate,
        "num_filters": args.num_filters,
        "filter_sizes": args.filter_sizes,
        "activation": args.activation,
        "fc_hidden_sizes": args.hidden_size,
        "dropout": args.dropout,
        "batch_norm": args.batch_norm.lower() == "true",
        "augmentation": args.augmentation.lower() == "true",
        "img_size": args.input_dim,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "base_dir": args.base_dir
    })
    
    return final_config, args.wandb_entity, args.wandb_project