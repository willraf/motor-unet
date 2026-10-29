"""
Script predict

Handles model inference.

Usage:
    $ python -m scripts.predict --config configs/predict.yaml
    $ python -m scripts.predict --config configs/predict.yaml --default configs/default.yaml

Author: 
    Will Raftery

Date:
    30/07/2025
"""

import logging
import time
import argparse

from omegaconf import OmegaConf

from src.data.setup import setup_adaptor
from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.models import setup_model

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main(cfg, overwrite=False):

    checkpoint_path = cfg.model.checkpoint
    plot_examples_flag = cfg.experiment.plot_examples
    n_sources = cfg.predict.num_sources
    n_plots = cfg.predict.num_plots


    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    save_dir = setup_save_dir(cfg, mode='inference', overwrite=overwrite)

    model = setup_model(cfg, device=device, mode='inference')
    load_checkpoint(checkpoint_path, model)

    # Loop through all files in the data directory
    if cfg.predict.data_dir is not None and cfg.predict.fname_list is not None:
        data_dir = Path(cfg.predict.data_dir)

        for data_idx, fname in enumerate(cfg.predict.fname_list):
            cfg.predict.data_path = str(data_dir / fname)
            logging.info(f'Predicting on file: {cfg.predict.data_path}')

            dataloaders, meta = setup_data(cfg, mode="inference")
            infer_loader = dataloaders['inference']
            emg = meta['full_emg']

            adaptor = setup_adaptor(cfg)

            predictions, avg_time = predict(model, 
                                  infer_loader,
                                  n_sources=n_sources, 
                                  adaptor=adaptor,
                                  save_dir=save_dir / f'{data_idx}',
                                  plot_examples=plot_examples_flag,
                                  n_plots=n_plots)

            # Save predictions and compute any metrics if necessary
            np.save(save_dir / f'{data_idx}' / 'predictions.npy', predictions)
            np.save(save_dir / f'{data_idx}' / 'raw.npy', emg)
    else:
        dataloaders, meta = setup_data(cfg, mode="inference")
        infer_loader = dataloaders['inference']
        emg = meta['full_emg']

        adaptor = setup_adaptor(cfg)

        predictions, avg_time = predict(model, 
                            infer_loader,
                            n_sources=n_sources, 
                            adaptor=adaptor,
                            save_dir=save_dir,
                            plot_examples=plot_examples_flag,
                            n_plots=n_plots)
        np.save(save_dir / 'predictions.npy', predictions)
        np.save(save_dir / 'raw.npy', emg)

    # Save predictions and compute any metrics if necessary


def predict(model, infer_loader, n_sources, adaptor, save_dir, plot_examples, n_plots):
    """
    Perform inference on a test set using the trained model.
    """
    start_time = time.time()

    num_batches = len(infer_loader)
    plot_interval = num_batches // n_plots

    predictions = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, meta) in enumerate(infer_loader):
            inputs = inputs.to(device)

            # Save original emg before peeling
            # Again, only works for batch size 1
            b, W, m, T = inputs.shape
            original_emg = adaptor.get_target_emg(inputs).clone()

            # Stores the predicted muaps in order matching with targets for plotting
            predicted_muaps = torch.zeros((b, n_sources, m, T), device=device)  # (n_sources, m, T)
            residuals = torch.zeros((b, n_sources, m, T), device=device)  # (n_sources, m, T)
            masks = torch.zeros((b, n_sources, m, T), device=device)  # (n_sources, m, T)

            sample_loss = 0.0

            # Iterate to get desired number of sources
            for source_idx in range(n_sources):
                # Forward pass
                output = model(inputs)

                predicted_muaps[:, source_idx] = output                        # (B, n, m, T)
                masks[:, source_idx] = adaptor.get_target_mask(inputs)         # (B, n, m, T)

                inputs, targets, meta = adaptor.peel_muap(inputs, targets, meta, output)
                residuals[:, source_idx] = adaptor.get_target_emg(inputs).clone()  # (B, n, m, T)


            if plot_examples and batch_idx % plot_interval == 0:
                # ica_sources = run_ica(emg_np, targets_np, num_components=n)
                # vis.plot_emg_muaps_residual_masks(emg, matched_targets, predicted_muaps, residuals=residuals, masks=masks, fpath=os.path.join(eval_dir, 'figures', f'test_{sample_num}.pdf'))
                vis.plot_emg_muaps_prediction(original_emg[0], residuals[0], predicted_muaps[0], fpath=save_dir / 'figures' / f'test_{batch_idx}.pdf')
                # vis.plot_ica_comparison(emg_np, ica_sources, targets_np, preds_np, fpath=os.path.join(eval_dir, f'ica_{sample_num}.pdf'))

            # Expand scale factors for broadcasting
            scale_factors = meta['scale_factors']                              # (B, num_sources)
            scale_factors = scale_factors.unsqueeze(-1).unsqueeze(-1)           # (B, num_sources, 1, 1)
            scale_factors = scale_factors[:, :-1]                               # Remove last scale factor as not used

            # Multiply back to original scale
            predicted_muaps = predicted_muaps * scale_factors
            residuals = residuals * scale_factors

            predictions.append(predicted_muaps.cpu().numpy())

    total_time = time.time() - start_time

    logging.info(f'Inference completed in {total_time:.4f} seconds')
    avg_time = total_time / num_batches
    logging.info(f'Average time per window: {avg_time:.4f} seconds')

    return np.concatenate(predictions, axis=0), avg_time        # (num_windows, n, m, T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    random_seed = cfg.experiment.random_seed
    set_seed(random_seed)

    main(cfg, overwrite=args.overwrite)