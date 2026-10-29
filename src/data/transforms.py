"""
Module transforms

Contains data transforms and collators for converting data into the right format for models.
Also include "Adaptors" which contain useful methods for handling model-specific data formats during training/evaluation
and apply these same transforms during iterative inference.

Transforms are kept in torch tensors so they can be resused during eval/inference loop.

Usage:
    from src.data.transforms import Normalise, AddHintMask, RestrictGroundTruth
    transform_list = [Normalise(), AddHintMask(), RestrictGroundTruth()]
    collator = UNetCollator()
    for t in transform_list:
        emg, muaps, meta = t(emg, muaps, meta)
    emg, muaps, meta = collator(emg, muaps, meta)

Authors: 
    Will Raftery

Date:
    05/09/2025
"""

from networkx import radius
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from typing import Any, Tuple   
from collections import defaultdict


class Normalise:
    """
    Normalize the final window of EMG data to be between -1 and 1. 
        Sources and context windows are normalised be the same factor to preserve relative contributions.

    Input shape: (W, m, T), (W, n, m, T), _
    Output shape: (W, m, T), (W, n, m, T), _
    """
    def __call__(self, emg, muaps, meta):
        device = emg.device
        if muaps is not None:
            muaps = muaps.to(device)

        scale_factor = torch.max(torch.abs(emg[-1])) 
        if scale_factor > 0:
            emg /= scale_factor
            if muaps is not None:
                muaps /= scale_factor
        
        # Save cumulative scale factors
        if "scale_factors" not in meta:
            meta["scale_factors"] = scale_factor.unsqueeze(0)  # shape (1,)
        else:
            meta["scale_factors"] = meta["scale_factors"].to(device)
            # Append new scale factor
            scale_factor *= meta["scale_factors"][-1]
            meta["scale_factors"] = torch.cat([
                meta["scale_factors"], 
                scale_factor.unsqueeze(0)
            ])

        return emg, muaps, meta
    

class AddHintMask:
    """
    Input shape: (W, m, T), (W, n, m, T), _
    Output shape: (W, 2, m, T), (W, n, m, T), _
    """
    def __call__(self, emg, muaps, meta):
        W, m, T = emg.shape
        device, dtype = emg.device, emg.dtype

        # Prevent the hint mask from being near edges of the window
        # pad = 16
        # emg_pad = emg.clone()
        # emg_pad[:, :, :pad] = 0.0
        # emg_pad[:, :, -pad:] = 0.0
        emg_pad = emg

        # Flatten each window to find max index
        flat_idx = torch.argmax(torch.abs(emg_pad.view(W, -1)), dim=1)
        rows, cols = torch.div(flat_idx, T, rounding_mode="floor"), flat_idx % T

        # Assign original EMG to first channel
        emg_out = torch.zeros((W, 2, m, T), dtype=dtype, device=device)
        emg_out[:, 0] = emg

        # Create hint masks for all windows at once
        mask_hint = torch.zeros_like(emg)
        mask_hint[torch.arange(W, device=device), rows, cols] = 1.0
        emg_out[:, 1] = mask_hint

        return emg_out, muaps, meta


class RestrictGroundTruth:
    """
    Restrict the ground truth to the MUAP with the highest absolute amplitude at the point indicated by the mask

    Input shape: (W, 2, m, T), (W, n, m, T), _
    Output shape: (W, 2, m, T), (W, m, T), _
    """
    def __call__(self, emg, muaps, meta):
        W, _, _, T = emg.shape
        _, n, m, _ = muaps.shape
        device, dtype = muaps.device, muaps.dtype

        if muaps is None:
            return emg, muaps, meta

        # Convert nan padding muaps to zero
        sources = torch.nan_to_num(muaps, nan=0.0)

        # Get hint indices per window
        mask = emg[:, 1]  # (W, m, T)
        hint_idx = torch.argwhere(mask == 1)       # 
        w_idx, rows, cols = hint_idx[:, 0], hint_idx[:, 1], hint_idx[:, 2]

        # Apply gaussian filter per window & MUAP
        muaps_smoothed = self.gaussian_filter(sources, sigma=3, radius=7, device=device, dtype=dtype)

        # Get top MUAP index at each window’s hint location
        top_idx = muaps_smoothed[w_idx, :, rows, cols].argmax(dim=1)
        top_muaps = muaps[torch.arange(W, device=device), top_idx, :, :]

        # Save remaining MUAPs in meta for peeling
        mask_idx = torch.ones((W, n), dtype=torch.bool, device=device)
        mask_idx[torch.arange(W), top_idx] = False
        remaining_muaps = muaps[mask_idx].reshape(W, n - 1, m, T)

        meta["remaining_muaps"] = remaining_muaps  # torch.Tensor

        return emg, top_muaps, meta
    
    def gaussian_filter(self, muaps, sigma: float, radius: int, device, dtype):
        W, n, m, T = muaps.shape

        # Create 1D Gaussian kernel
        kernel = self.gaussian_kernel1d(sigma, radius, device=device, dtype=dtype)

        # Smooth squared MUAPs across time
        muaps_sq = muaps ** 2
        muaps_sq = muaps_sq.view(W * n * m, 1, T)
        muaps_smoothed = torch.nn.functional.conv1d(muaps_sq, kernel, padding=radius)
        muaps_smoothed = muaps_smoothed.view(W, n, m, T)
        return muaps_smoothed

    
    def gaussian_kernel1d(self, sigma: float, radius: int, device=None, dtype=None):
        x = torch.arange(-radius, radius+1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel.view(1, 1, -1)  # (out_channels, in_channels, kernel_size)

    

class UNetCollator:
    """
    Input shapes:
        emg: (1, 2, m, T)
        muaps: (1, m, T)

    Output shapes:
        emg: (2, m, T)
        muaps: (m, T)
    """

    def __call__(self, 
                 emg: torch.Tensor, 
                 muaps: torch.Tensor | None, 
                 meta: Any
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # # Squeeze window dimension
        emg = torch.squeeze(emg, axis=0)    # (2, m, T)
        muaps = torch.squeeze(muaps, axis=0) if muaps is not None else None  # (m, T)
        return emg, muaps, meta


class TransformerCollator:
    """
    Input shapes:
        emg: (W, 2, m, T)
        muaps: (W, m, T)

    Output shapes:
        emg: (W, 2, m, T)
        muap: (m, T)
    """

    def __call__(self, 
                 emg: torch.Tensor,
                 muaps: torch.Tensor | None, 
                 meta: Any):
        muaps = muaps[-1, :, :] if muaps is not None else None  # (m, T)
        return emg, muaps, meta


class UNetAdaptor:
    """
    Contains useful methods for handling data from UNet models.

    Used for funcitonality which requires knowledge of the data format, 
        so this can be abstracted away from the main train/eval code.
        For example, peeling muaps and transforming data, getting visualisation data etc.
    """
    def get_visualisation_bundle(self, inputs, targets, outputs):
        """
        Args:
            inputs: (B, 2, m, T) - EMG + hint mask
            targets: (B, m, T) - target MUAP
            outputs: (B, m, T) - predicted MUAP

        Return:
            - emg_plot: (m, T)
            - gt_plot: (2, m, T)
            - predicted_plot + mask: (2, m, T)
        """
        # Get first element of batch 
        inputs = inputs[0]                                            # (2, m, T)
        targets = targets[0]                                          # (m, T)
        outputs = outputs[0]                                          # (m, T)

        predicted_plot = outputs.unsqueeze(0)                            # (1, m, T)
        emg = inputs[0]                                                # (m, T)
        max_val_mask = inputs[1].unsqueeze(0)                            # (1, m, T)  
        predicted_plot = torch.cat([predicted_plot, max_val_mask], dim=0)  # (2, m, T)
        gt_plot = torch.full_like(predicted_plot, float('nan'))         # (2, m, T)
        gt_plot[0] = targets                   
        return emg, gt_plot, predicted_plot


    def get_target_mask(self, inputs):
        """
        Extract the target mask from the inputs.

        Args:
            inputs: (B, 2, m, T) - EMG + hint mask

        Returns:
            target_mask: (B, m, T) - Target mask
        """
        return inputs[:, 1, :, :]


    def get_target_emg(self, inputs):
        """
        Extract the target EMG from the inputs.

        Args:
            inputs: (B, 2, m, T) - EMG + hint mask

        Returns:
            target_emg: (B, m, T) - Target EMG
        """
        return inputs[:, 0, :, :]
    

    def peel_muap(self, inputs, targets, meta, outputs):
        """
        Strip the predicted MUAP from EMG

        Args:
            emg: (B, 2, m, T) - EMG + hint mask
            muaps: (B, m, T) - target MUAP
            meta: dict - metadata
            output: (B, m, T) - predicted MUAP

        Returns:
            (B, 2, m, T) - EMG with the predicted MUAP subtracted
            (B, m, T) - next target MUAP
            (B, dict) - metadata, including remaining MUAPs
        """
        emg = inputs[:, 0, :, :]  # (B, m, T) - Extract the EMG from the input
        emg = emg - outputs   # (B, m, T) - Subtract the predicted MUAPs
        inputs = emg.unsqueeze(1)  # (B, 1, m, T) - Reintroduce the window dimension
        emg_batch, muaps_batch = [], []
        meta_batches = defaultdict(list)
        for i in range(inputs.shape[0]):
            emg_i = inputs[i]
            # Meta is a dict
            meta_i = {key: val[i] for key, val in meta.items()}
            muaps_i = meta_i.get('remaining_muaps', None)       # (W, n-1, m, T)

            emg_i, muaps_i, meta_i = Normalise()(emg_i, muaps_i, meta_i)
            emg_i, muaps_i, meta_i = AddHintMask()(emg_i, muaps_i, meta_i)
            if muaps_i is not None:
                emg_i, muaps_i, meta_i = RestrictGroundTruth()(emg_i, muaps_i, meta_i)

            emg_i, muaps_i, meta_i = UNetCollator()(emg_i, muaps_i, meta_i)

            emg_batch.append(emg_i)
            muaps_batch.append(muaps_i)

            # Save meta fields
            for k, v in meta_i.items():
                meta_batches[k].append(v)

        # Stack batch back together
        emg = torch.stack(emg_batch, dim=0)
        muaps = torch.stack(muaps_batch, dim=0) if muaps_batch[0] is not None else None
        # Batched meta
        meta = {}
        for k, vlist in meta_batches.items():
            # Try to stack if possible
            if isinstance(vlist[0], torch.Tensor):
                meta[k] = torch.stack(vlist, dim=0)
            else:
                meta[k] = vlist  # keep as list if not tensor
        return emg, muaps, meta


class TransformerAdaptor:

    def get_visualisation_bundle(self, inputs, targets, outputs):
        """
        Args:
            inputs: ((B, W, 2, m, T)) - context emg, target emg + hint mask
            targets: (B, m, T) - target MUAP
            outputs: (B, m, T) - predicted MUAP

        Return:
            - emg_plot: (m, T)
            - gt_plot: (2, m, T)
            - predicted_plot + mask: (2, m, T)
        """
        predicted_plot = outputs[0].unsqueeze(0)                            # (1, m, T)
        emg = inputs[0, -1, 0, :, :]                                            # (m, T)
        max_val_mask = inputs[0, -1, 1, :, :]

        # max_val_mask = emg.amax(dim=(0, 1), keepdim=True) == emg.abs()
        predicted_plot = torch.cat([predicted_plot, max_val_mask.int().unsqueeze(0)], dim=0)

        gt_plot = torch.full_like(predicted_plot, float('nan'))         # (2, m, T)
        gt_plot[0] = targets[0]

        emg_plot = emg
        return emg_plot, gt_plot, predicted_plot  


    def get_target_emg(self, inputs):
        """
        Extract the target EMG from the inputs.

        Args:
            inputs: (B, W, 2, m, T) - context emg, target emg + hint mask

        Returns:
            target_emg: (B, m, T) - Target EMG
        """
        return inputs[:, -1, 0, :, :]


    def get_target_mask(self, inputs):
        """
        Extract the target mask from the inputs.

        Args:
            inputs: (B, W, 2, m, T) - context emg, target emg + hint mask

        Returns:
            target_mask: (B, m, T) - Target mask
        """
        return inputs[:, -1, 1, :, :]


    def peel_muap(self, inputs, targets, meta, outputs):
        """
        Strip the predicted MUAP from EMG

        Args:
            inuputs: (B, W, 2, m, T) - EMG + hint mask
            targets: (B, W, m, T) - target MUAP
            meta: dict - metadata
            outputs: (B, m, T) - predicted MUAP

        Returns:
            (B, W, 2, m, T) - EMG with the predicted MUAP subtracted
            (B, W, m, T) - next target MUAP
            (B, dict) - metadata, including remaining MUAPs
        """
        emg = inputs[:, -1, 0, :, :]  # (B, m, T) - Extract the EMG from the input
        emg = emg - outputs   # (B, m, T) - Subtract the predicted MUAPs
        inputs[:, -1, 0, :, :] = emg  # (B, W, 2, m, T) - update the target window emg
        inputs = inputs[:, :, 0, :, :] # (B, W, m, T) - Remove the hint mask dimension

        emg_batch, muaps_batch = [], []
        meta_batches = defaultdict(list)
        for i in range(inputs.shape[0]):
            emg_i = inputs[i]
            # Meta is a dict
            meta_i = {key: val[i] for key, val in meta.items()}
            muaps_i = meta_i.get('remaining_muaps', None)       # (W, n-1, m, T)

            emg_i, muaps_i, meta_i = Normalise()(emg_i, muaps_i, meta_i)
            emg_i, muaps_i, meta_i = AddHintMask()(emg_i, muaps_i, meta_i)
            if muaps_i is not None:
                emg_i, muaps_i, meta_i = RestrictGroundTruth()(emg_i, muaps_i, meta_i)

            emg_i, muaps_i, meta_i = TransformerCollator()(emg_i, muaps_i, meta_i)

            emg_batch.append(emg_i)
            muaps_batch.append(muaps_i)

            # Save meta fields
            for k, v in meta_i.items():
                meta_batches[k].append(v)

        # Stack batch back together
        emg = torch.stack(emg_batch, dim=0)
        muaps = torch.stack(muaps_batch, dim=0) if muaps_batch[0] is not None else None
        # Batched meta
        meta = {}
        for k, vlist in meta_batches.items():
            # Try to stack if possible
            if isinstance(vlist[0], torch.Tensor):
                meta[k] = torch.stack(vlist, dim=0)
            else:
                meta[k] = vlist  # keep as list if not tensor
        return emg, muaps, meta
    

if __name__ == "__main__":
    emg = np.random.randn(1, 12, 64)
    muaps = np.random.randn(1, 5, 12, 64)
    meta = {}

    transform = AddHintMask()
    emg, muaps, meta = transform(emg, muaps, meta)

    transform = RestrictGroundTruth()
    emg, muaps, meta = transform(emg, muaps, meta)
