""" Module setup

Provides functions to set up the dataset and dataloaders automatically

Usage:
    from src.data.setup import setup_data
    from src.data import setup_data

    dataloaders, scheduler = setup_data(cfg, mode="train")

Authors:
    Will Raftery

Date:
    31/07/2025
"""

from omegaconf import OmegaConf
from typing import Optional

import numpy as np

from src.data.curriculum_scheduler import CurriculumScheduler
from src.data.generators import SingleSpikeDataGenerator, ConvolvedDataGenerator, RealDataGenerator
from src.data.transforms import AddHintMask, Normalise, RestrictGroundTruth, UNetCollator, TransformerCollator, UNetAdaptor, TransformerAdaptor
from src.data.general_dataset import GeneralDataset

from torch.utils.data import DataLoader, random_split

import logging

def setup_adaptor(cfg: OmegaConf):

    model_name = cfg.model.name.lower()

    if model_name == "unet2d" or model_name == "unet":
        adaptor = UNetAdaptor()
    elif model_name == "transformer":
        adaptor = TransformerAdaptor()
    else:
        raise ValueError(f"Model {model_name} not recognised")
    
    return adaptor


def setup_data(cfg: OmegaConf, emg: Optional[np.ndarray] = None, mode: str = "train"):
    """
    Returns:
        dataloaders (dict): dictionary of dataloaders for the specified mode
        meta (dict): dictionary of metadata, e.g. scheduler, 
    """

    model_name = cfg.model.name.lower()

    dataloaders = {}
    meta = {}
    scheduler = None


    # Create data generator
    if mode in ['train', 'eval']:
        generator = SingleSpikeDataGenerator(cfg, mode=mode)
    elif mode == "inference":
        generator = RealDataGenerator(cfg, emg=emg, mode=mode)
        meta['full_emg'] = generator.emg
    elif mode == "spike_train_eval":
        generator = ConvolvedDataGenerator(cfg, mode=mode)
    else:
        raise ValueError(f"Mode {mode} not recognised")
    
    # Transforms and collators
    transforms = []
    transforms.append(Normalise())
    # transforms.append(CircularPad(padding_size))
    transforms.append(AddHintMask())
    if mode in ['train', 'eval']:  # only for supervised tasks
        transforms.append(RestrictGroundTruth())

    # Collator
    if model_name == "unet2d" or model_name == "unet":
        collator = UNetCollator()
    elif model_name == "transformer":
        collator = TransformerCollator()
    else:
        raise ValueError(f"Model {model_name} not recognised")
    
    # Dataset
    dataset = GeneralDataset(
        cfg=cfg,
        generator=generator,
        transforms=transforms,
        collator=collator,
    )

    # Dataloaders and scheduler
    if mode == 'train':

        val_split = cfg.training.val_split
        train_batch_size = cfg.training.batch_size
        val_batch_size = cfg.validation.batch_size
        num_workers = cfg.data.num_workers
        curriculum_flag = cfg.data.curriculum.enabled

        n_total = len(dataset)
        n_val = int(val_split * n_total)
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
            
        # Create DataLoaders for training and validation
        dataloaders['train'] = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
            num_workers=num_workers
        )
        dataloaders['val'] = DataLoader(
            val_dataset, 
            batch_size=val_batch_size, 
            shuffle=True,
            num_workers=num_workers
        )

        if curriculum_flag:
            scheduler = CurriculumScheduler(
                dataset=dataset,
                milestones=cfg.data.curriculum.milestones,
                parameters=cfg.data.curriculum.parameters
            )
            meta['scheduler'] = scheduler

    elif mode == 'eval':
        dataloaders['eval'] = DataLoader(
            dataset, 
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers
        )
    
    elif mode == 'inference':
        dataloaders['inference'] = DataLoader(
            dataset, 
            batch_size=cfg.predict.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers
        )

    elif mode == 'spike_train_eval':
        dataloaders['spike_train_eval'] = DataLoader(
            dataset, 
            batch_size=cfg.spike_train_eval.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers
        )
    
    return dataloaders, meta

  