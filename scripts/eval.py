"""
Script eval

Handles model testing/evaluation.

Usage:
    $ python -m scripts.eval --config configs/eval.yaml
    $ python -m scripts.eval --config configs/eval.yaml --default configs/default.yaml

Author: 
    Will Raftery

Date:
    30/07/2025
"""

import argparse
from pathlib import Path
import logging
import csv
import time

from omegaconf import OmegaConf

from typing import Union, Optional, Callable

import pandas as pd

from src.utils.utils import *
import src.utils.visualisation as vis
import src.utils.metrics as metrics

from src.data import setup_data, setup_adaptor
from src.models import setup_model

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def save_results(save_dir, results_dict):
    """
    Save test results (dict of metrics) to a CSV file
    """
    test_results_fpath = Path(save_dir) / "results.csv"
    file_exists = test_results_fpath.exists()

    df_rows = []
    for k, v in list(results_dict.items()):
        if isinstance(v, (list, tuple, np.ndarray)):
            results_dict.pop(k)  # safe now
            for i, val in enumerate(v):
                df_rows.append({'iteration': i, k: val})
                
    if df_rows:  # save per-iteration values
        df = pd.DataFrame(df_rows)
        df.to_csv(Path(save_dir) / "results_per_iteration.csv", index=False)

    with open(test_results_fpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_dict)

def main(cfg, overwrite=False):

    random_seed = cfg.experiment.random_seed
    checkpoint_path = cfg.model.checkpoint
    plot_examples_flag = cfg.experiment.plot_examples
    num_plots = cfg.evaluation.num_plots
    num_sources = cfg.evaluation.num_sources

    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Set the seed for reproducibility
    set_seed(random_seed)

    save_dir = setup_save_dir(cfg, overwrite=overwrite, mode="eval")

    dataloaders, _ = setup_data(cfg, mode="eval")
    test_loader = dataloaders['eval']

    adaptor = setup_adaptor(cfg)

    model = setup_model(cfg, device=device, mode='eval')
    load_checkpoint(checkpoint_path, model)

    loss_function = setup_loss_function(cfg)
    
    # Test model
    results = test(model, 
                   test_loader, 
                   loss_function, 
                   n_sources=num_sources, 
                   adaptor=adaptor,
                   save_dir=save_dir, 
                   plot_examples=plot_examples_flag, 
                   n_plots=num_plots, 
    )
    logging.info(f"Test Loss: {results['loss']:.6f}")
    save_results(save_dir, results)

    return results


def test(model, test_loader, loss_function, n_sources, adaptor, save_dir, plot_examples, n_plots):
    """
    Perform inference on a test set using the trained model.
    """
    totals = dict(loss=0.0, accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, roa=0.0)
    num_batches = len(test_loader)
    plot_interval = num_batches // n_plots
    if plot_interval == 0:
        plot_interval = 1

    # Record per iteration metrics
    total_mse_per_iter = torch.zeros(n_sources).to(device)
    total_baseline_per_iter = torch.zeros(n_sources).to(device)  
    total_time = 0.0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, meta) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Save original emg before peeling
            # Again, only works for batch size 1
            b, m, T = targets.shape
            original_emg = adaptor.get_target_emg(inputs).clone()

            # Stores the predicted muaps in order matching with targets for plotting
            preds_batch = []
            targets_batch = []
            residuals_batch = []
            masks_batch = []
            sample_loss = 0.0

            start_time = time.time()

            # Iterate to get desired number of sources
            for source_idx in range(n_sources):
                
                # Forward pass
                output = model(inputs)

                preds_batch.append(output)                  # List[(b, m, T)]
                targets_batch.append(targets)               # List[(b, m, T)]
                masks_batch.append(adaptor.get_target_mask(inputs)) 

                loss = loss_function(output, targets)

                sample_loss += loss.item()

                zero_pred = torch.zeros_like(output)
                baseline = loss_function(zero_pred, targets)

                total_baseline_per_iter[source_idx] += baseline.item()
                total_mse_per_iter[source_idx] += loss.item()

                inputs, targets, meta = adaptor.peel_muap(inputs, targets, meta, output)
                residuals_batch.append(adaptor.get_target_emg(inputs))

            preds_batch = torch.stack(preds_batch, dim=1)     # (b, n_sources, m, T)
            targets_batch = torch.stack(targets_batch, dim=1) # (b, n_sources, m, T)
            masks_batch = torch.stack(masks_batch, dim=1)     # (b, n_sources, m, T)
            residuals_batch = torch.stack(residuals_batch, dim=1) # (b, n_sources, m, T)
    
            end_time = time.time()
            total_time += (end_time - start_time)

            totals['loss'] += sample_loss / n_sources  # Average loss over sources
            acc, prec, rec, f1, roa = metrics.spike_metrics(preds_batch, targets_batch)

            for key, val in zip(["accuracy", "precision", "recall", "f1", "roa"], [acc, prec, rec, f1, roa]):
                totals[key] += val

            if plot_examples and batch_idx % plot_interval == 0:
                # ica_sources = run_ica(emg_np, targets_np, num_components=n)
                # vis.plot_emg_muaps_residual_masks(emg, matched_targets, predicted_muaps, residuals=residuals, masks=masks, fpath=os.path.join(eval_dir, 'figures', f'test_{sample_num}.pdf'))
                vis.plot_emg_muaps_prediction(original_emg[0], targets_batch[0], preds_batch[0], fpath=save_dir / 'figures' / f'test_{batch_idx}.pdf')
                # vis.plot_ica_comparison(emg_np, ica_sources, targets_np, preds_np, fpath=os.path.join(eval_dir, f'ica_{sample_num}.pdf'))
    
    avg_baseline = (total_baseline_per_iter / num_batches).cpu().numpy() 
    avg_mse = (total_mse_per_iter / num_batches).cpu().numpy()
    avg_nmse = avg_mse / avg_baseline

    avg_time = total_time / num_batches
    avg_time_per_source = total_time / (num_batches * n_sources)

    logging.info(f'Average time per window: {avg_time:.4f} seconds')
    logging.info(f'Average time per source: {avg_time_per_source:.4f} seconds')

    if plot_examples:
        vis.plot_nmse(avg_nmse, fpath=save_dir / 'figures' / 'avg_nmse_per_iteration.pdf')

    results = {k: v / num_batches for k, v in totals.items()}

    logging.info(
        "\nTest set: " +
        " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in results.items()])
    )
    
    results.update({
        'avg_mse_per_iteration': avg_mse,
        'avg_baseline_per_iteration': avg_baseline,
        'avg_nmse_per_iteration': avg_nmse,
        'avg_time_per_window': avg_time,
        'avg_time_per_source': avg_time_per_source,
    })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)

    random_seed = cfg.experiment.random_seed
    set_seed(random_seed)

    main(cfg, overwrite=args.overwrite)
