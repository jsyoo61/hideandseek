import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

import hideandseek as hs

# Setup logging
log = logging.getLogger(__name__)

# Define main function using the Hydra decorator
@hydra.main(config_path='conf', config_name='train1', version_base='1.2')
def main(cfg: DictConfig) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'device: {device}')

    # Generate synthetic data (200 random samples)
    x = torch.rand(200, 1)
    y = 5 * x + 2

    # Create a linear network (1 input, 1 output)
    network = nn.Linear(1, 1)
    network.to(device)

    # Create dataset using synthetic data
    dataset = torch.utils.data.TensorDataset(x, y)

    # Define loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Create a path for saving the network (if necessary)
    path_dict = {'network': Path('network')}
    log.info(f'CWD: {os.getcwd()}')

    # Define metrics for evaluation (optional)
    metrics = {
        'accuracy': hs.E.accuracy_score,  # Define or replace this with a relevant metric
    }

    # Prepare the training parameters
    kwargs = {
        'network': network,
        'train_dataset': dataset,
        'cfg_train': OmegaConf.to_container(cfg.train['update'], resolve=True),
        'criterion': criterion,
        'network_dir': path_dict['network'],
        'cfg_val': OmegaConf.to_container(cfg.train.validation, resolve=True),
        'val_dataset': dataset,  #Replace this with actual validation data
        'val_metrics': metrics,
        'verbose': False,  # Optional: Set to True for detailed logs
        'amp': cfg.amp  # Mixed precision training setting
    }

    # Instantiate the trainer using hideandseek's Trainer class
    trainer = hs.Trainer(**kwargs)

    # Start training (using the configured epochs in `cfg.train`)
    trainer.train()

    # Optionally train for additional epochs or steps if needed
    trainer.train(5)  # Train for an additional 5 epochs
    trainer.train(epoch=5)  # Another way to train for 5 epochs
    trainer.train(step=500)  # Train for 500 steps instead of epochs

    # Move the trained model back to CPU if necessary
    trainer.network.cpu()

if __name__ == "__main__":
    main()