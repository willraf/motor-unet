"""
Script train

Handles model training and validation.

Usage:
    $ python -m scripts.train --config configs.train.yaml
    $ python -m scripts.train --config configs.train.yaml --default configs/default.yaml

Author: 
    Will Raftery

Date:
    30/07/2025
"""

import argparse
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
import time
from pathlib import Path
import csv


# TODO remove generic utils import *
from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data, setup_adaptor
from src.models import setup_model

import torch

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def save_results(save_dir, epoch, train_loss, val_loss, best_val_loss):
    """
    Append training results to file
    """
    train_results_fpath = Path(save_dir) / "results.csv"
    
    with open(train_results_fpath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, best_val_loss])


def main(cfg, overwrite=False):

    patience = cfg.training.patience
    log_interval = cfg.logging.log_interval
    num_epochs = cfg.training.epochs
    plot_examples_flag = cfg.experiment.plot_examples

    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    save_dir = setup_save_dir(cfg, overwrite=overwrite, mode="train")

    dataloaders, meta = setup_data(cfg, mode="train")
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    scheduler = meta.get('scheduler', None)

    adaptor = setup_adaptor(cfg)

    model = setup_model(cfg, device=device, mode="train")

    optimizer = setup_optimiser(cfg, model)

    loss_function = setup_loss_function(cfg)

    results = {
        "epochs": [],   # Each epoch has different data (more like "meta-batches" after which validation is performed)
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": []
    }

    # For early stopping and saving best model
    best_val_loss = float('inf')

    counter = 0
    
    # Train-test loop
    for epoch in tqdm(range(num_epochs)):
        # Curriculum learning: update the dataset parameters
        if scheduler:
            scheduler.update(epoch)
        
        start_time = time.time()

        train_loss = train(model, 
            train_loader, 
            optimizer, 
            loss_function, 
            epoch,
            log_interval=log_interval,
        )
        
        val_loss = validate(model, 
            val_loader, 
            loss_function,
            plot_examples=plot_examples_flag,                
            epoch=epoch,
            save_dir=save_dir,
            adaptor=adaptor,
        )

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / "checkpoints" / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break


        # For plotting
        results["epochs"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss) 
        results["best_val_loss"].append(best_val_loss)

        elapsed_time = time.time() - start_time 

        logging.info(f'====> Epoch: {epoch} Average train loss: {train_loss:.4f}, Average val loss: {val_loss:.4f}, Elapsed time: {elapsed_time:.2f} seconds')
        
        # Save results to file
        save_results(save_dir, epoch, train_loss, val_loss, best_val_loss)
        
        if plot_examples_flag:
            vis.plot_loss_curves(results["train_loss"], results["val_loss"], results["epochs"], fpath= save_dir / "figures" / "loss_curves.pdf")

    return results['best_val_loss'][-1]  # Return the best validation loss from the last epoch


def train(model, train_loader, optimizer, loss_function, epoch, log_interval):
    """
    A utility function that performs a basic training loop.

    For each batch in the training set, fetched using `train_loader`:
        - Zeroes the gradient used by `optimizer`
        - Performs forward pass through `model` on the given batch
        - Computes loss on batch
        - Performs backward pass
        - `optimizer` updates model parameters using computed gradient

    Prints the training loss on the current batch every `log_interval` batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    start = time.time()

    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        print(f"Batch {batch_idx+1}/{num_batches} -- Time elapsed: {time.time() - start:.2f} seconds, device: {device}", end="\r")

        # Send batch to the device we are using
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zeroes the gradient used by `optimizer`
        optimizer.zero_grad()

        outputs = model(inputs)

        # NOTE  Use mean reduction
        #   Doesnt really matter as long as left consistent for training (loss scaled by constant factor)
        loss = loss_function(outputs, targets) #, reduction="mean")

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if log_interval is not None:
            # Prints the training loss on the current batch every `log_interval` batches.
            if batch_idx % log_interval == 0:
                logging.info(
                    "Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
                        epoch,
                        batch_idx,
                        loss.item(),
                    )
                )
        
        torch.cuda.empty_cache()

    # Note that the average loss is computed differently to the test funciton, 
    # but they both compute the same thing: the avereage loss over all datapoints in the dataset
    return total_loss / num_batches


def validate(model, test_loader, loss_function, plot_examples, epoch, save_dir, adaptor):
    """
    A utility function to compute the loss and accuracy on a test set by
    iterating through the test set using the provided `test_loader` and
    accumulating the loss and accuracy on each batch.       
    """
    val_loss = 0.0

    num_batches = len(test_loader)

    model.eval()
    with torch.no_grad():
        for (inputs, targets, _) in test_loader:
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)

            loss = loss_function(outputs, targets) #, reduction="mean")
            
            val_loss += loss.item()

    if plot_examples:
        emg_plot, gt_plot, prediction_plot = adaptor.get_visualisation_bundle(inputs, targets, outputs)
        vis.plot_emg_muaps_prediction(emg_plot, gt_plot, prediction_plot, fpath=save_dir / "figures" / f'val_epoch_{epoch}.pdf')

    # Divide by total number of samples to get average loss
    val_loss /= num_batches

    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    random_seed = cfg.experiment.random_seed
    set_seed(random_seed)   
    
    main(cfg, overwrite=args.overwrite)
