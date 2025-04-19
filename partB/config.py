"""
Configuration module for Part B: Pretrained ResNet-50 fine-tuning.
Handles default hyperparameters and command-line parsing.
"""
import argparse

# Default settings
DEFAULT_CONFIG = {
    "base_dir": "inaturalist_12K",  # dataset root with train/ and val/
    "batch_size": 32,                # mini-batch size
    "learning_rate": 1e-4,           # initial learning rate
    "augmentation": False,            # apply data augmentation
    "strategy": 1,                   # fine-tuning strategy (1=freeze backbone)
    "img_size": 224,                 # input image size
    "num_epochs": 10                 # number of training epochs
}


def get_config():
    """
    Parse CLI arguments and merge with defaults.
    Returns:
        config (dict): Combined configuration dictionary.
    """
    parser = argparse.ArgumentParser(
        description='Fine-tune ResNet-50 on iNaturalist-12K'
    )
    parser.add_argument('--wandb_entity', '-we', type=str, default=None,
                        help='Weights & Biases entity/account')
    parser.add_argument('--wandb_project', '-wp', type=str, default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--base_dir', '-br', type=str,
                        default=DEFAULT_CONFIG['base_dir'],
                        help='Path to dataset root (train/ and val/)')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=DEFAULT_CONFIG['batch_size'],
                        help='Mini-batch size')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=DEFAULT_CONFIG['learning_rate'],
                        help='Initial learning rate')
    parser.add_argument('--augmentation', '-au', choices=['true', 'false'],
                        default=str(DEFAULT_CONFIG['augmentation']).lower(),
                        help='Enable data augmentation')
    parser.add_argument('--strategy', '-s', type=int,
                        default=DEFAULT_CONFIG['strategy'],
                        help='Fine-tuning strategy (1=freeze backbone)')
    parser.add_argument('--num_epochs', '-e', type=int,
                        default=DEFAULT_CONFIG['num_epochs'],
                        help='Number of training epochs')
    args = parser.parse_args()

    # Merge parsed args with defaults
    config = DEFAULT_CONFIG.copy()
    config.update({
        'wandb_entity': args.wandb_entity,
        'wandb_project': args.wandb_project,
        'base_dir': args.base_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'augmentation': args.augmentation.lower() == 'true',
        'strategy': args.strategy,
        'num_epochs': args.num_epochs
    })
    return config
