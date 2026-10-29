from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import shutil
import csv

from omegaconf import OmegaConf

from typing import Union, Optional, Callable

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


def load_config(user_path: Path, default_path: Path) -> OmegaConf:
    """
    Load and merge user and default configuration files.

    Args:
        user_path (Path): Path to the user configuration file.
        default_path (Path): Path to the default configuration file.

    Returns:
        OmegaConf: Merged configuration object.
    """
    user_config = OmegaConf.load(user_path)
    default_config = OmegaConf.load(default_path)
    merged_config = OmegaConf.merge(default_config, user_config)
    return merged_config


def set_seed(seed: int):
    """
    Set seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_experiment_directory(cfg: OmegaConf) -> Path:
    """Create/get a directory for the experiment with a structured naming convention.
    
    Args:
        config (Config): Configuration object

    Returns:
        Path: Path to the experiment directory
    """
    save_flag = cfg.experiment.save
    experiment_name = cfg.experiment.name
    model_name = cfg.model.name

    if save_flag:
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"{timestamp}_{model_name}"
        experiment_dir = Path("experiments") / experiment_name
    else:
        experiment_dir = Path("experiments") / "TEMP"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def setup_logger(experiment_dir: Union[str, Path]): 
    """
    Setup logger for experiment
    """
    log_dir = Path(experiment_dir) / "logs"
    # This line should be redundant
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "experiment.log"),
            logging.StreamHandler()
        ]
    )


def change_log_file(log_fpath: Union[str, Path]):

    logger = logging.getLogger()  # Get the root logger
    
    # Remove existing file handlers
    for handler in logger.handlers[:]:  
        if isinstance(handler, logging.FileHandler):  
            logger.removeHandler(handler)
            handler.close()

    # Add new file handler
    file_handler = logging.FileHandler(log_fpath)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"Log file changed to: {log_fpath}")


def setup_loss_function(cfg: OmegaConf = None) -> Callable:
    """
    Create loss function from config file

    TODO: Move to src/losses/setup.py

    Args:
        config (dict): The configuration dictionary

    Returns:
        loss_function (function): The loss function
    """
    loss_type = cfg.loss_function

    if loss_type == 'mse':
        loss_function = F.mse_loss
    else:
        raise ValueError(f"Loss function {loss_type} not recognised")

    return loss_function


def setup_optimiser(cfg: OmegaConf, model: nn.Module) -> optim.Optimizer:
    """
    Create optimiser from config file
    """
    optimizer_name = cfg.training.optimizer
    learning_rate = cfg.training.learning_rate
    momentum = cfg.training.momentum
    weight_decay = cfg.training.weight_decay

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=momentum
        )
        logging.info("Using SGD optimizer")
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logging.info("Using Adam optimizer")
    elif optimizer_name == 'adamw':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logging.info("Using Adam optimizer")
    else:
        raise ValueError("Optimizer not recognised")

    return optimizer


def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    val_loss: float,
                    checkpoint_path: Union[str, Path]):
    """
    Save model and optimizer state as a checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        val_loss (float): The validation loss at this epoch.
        checkpoint_path (str): Path to save the checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, Path(checkpoint_path))
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Union[str, Path],
                    model: nn.Module,
                    optimizer: Optional[optim.Optimizer] = None):
    """
    Load a checkpoint from a file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into. Defaults to None.

    Returns:
        None
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_file():
        logging.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f"Loaded optimizer state from '{checkpoint_path}'")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")


def setup_save_dir(cfg: OmegaConf,
                   mode: str,
                   overwrite: bool = False):
    """
    Setup model and results saving

    If save is False, results are still saved to a TEMP directory and will be overwritten each run

    Args:
        cfg (OmegaConf): Configuration object
        overwrite (bool): Whether to overwrite existing directories
        mode (str): Mode of operation, either "train" "eval" or "predict"

    Returns:
        save_dir (Path): Path to the directory where results will be saved
    """
    # Create experiment directory
    experiment_dir = Path(get_experiment_directory(cfg))
    save_dir = experiment_dir / mode 

    # Handle existing training directory
    if save_dir.exists():
        logging.warning(f"Directory {save_dir} already exists.")

        if overwrite:
            logging.info("Overwriting existing directory.")
            shutil.rmtree(save_dir)
        else:
            response = input(f"Directory '{save_dir}' already exists. Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                logging.info("Aborting run.")
                exit(0)
            else:
                shutil.rmtree(save_dir)

    # Create necessary directories
    save_dir.mkdir(parents=True, exist_ok=False)
    (save_dir / "figures").mkdir(exist_ok=True)

    # Save logger to file
    change_log_file(save_dir / f"{mode}.log")

    # Save a copy of the config file
    OmegaConf.save(cfg, save_dir / "config.yaml")

    if mode == "train":

        (save_dir / "checkpoints").mkdir(exist_ok=True)

        # Create file to save training results
        with (save_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss"])
    
    return save_dir
