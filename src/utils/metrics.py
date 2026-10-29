"""
Module metrics

Contains functions to calculate metrics for EMG decomposition.

Usage:
    import src.utils.metrics as metrics

    accuracy = metrics.accuracy(predicted_muaps, true_muaps)

Authors:

Date:
    29/04/2025

TODO: 
    - Fix issue where a tiny amount of amplitude in the prediction counts as a spike, leading to higher tp that perhaps there should be
"""

from typing import List
import torch
import numpy as np
import scipy 
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
from scipy.signal import istft


class SVDMetric:

    def __init__(self, muaps: List[np.ndarray]):
        """
        Initialize the SVDMetric with a precomputed SVD basis.
        
        Args:
            svd_basis (np.ndarray): Precomputed SVD basis of shape (n, m).
        """
        self.svd_basis = self._get_svd_basis(muaps)

    def __call__(self, emg: np.ndarray, predicted_sources: np.ndarray) -> float:
        return svd_score(emg, predicted_sources, self.svd_basis)

    def _get_svd_basis(self, muaps: List[np.ndarray]) -> np.ndarray:
        """
        Compute the SVD basis from a list of MUAPs.

        Code taken from Kostia
        
        Args:
            muaps (List[np.ndarray]): List of MUAPs, each of shape (m, T).
        
        Returns:
            np.ndarray: SVD basis of shape (m, len(muaps)).
        """
        svd_basis = np.zeros([12, len(muaps)])
        for i, m in enumerate(muaps):
            u, s, v = np.linalg.svd(m, full_matrices=False)
            j = np.argmax(np.abs(u[:, 0]))
            svd_basis[:, i] = u[:, 0] * np.sign(u[j, 0])

        return svd_basis


def svd_score(emg: np.ndarray, predicted_sources: np.ndarray, svd_basis: np.ndarray) -> float:
    """
    Calculate the SVD score for the predicted sources

    Code taken from Kostia
    
    Args:
        emg (np.ndarray): shape (m, T_total). tshe original EMG signal.
        predicted_sources (np.ndarray): shape (num_windows, n, m, T) Predicted sources from the model.
        svd_basis (np.ndarray): ???

    Returns:
        float: SVD score.
    """
    
    # svd_basis metric
    predictions_concat = np.concatenate(predicted_sources, axis=0)
    basis_predictions = np.zeros([12, len(predictions_concat)])
    for i, muaps_prediction in enumerate(predictions_concat):
        u,s,v = np.linalg.svd(muaps_prediction, full_matrices=False)
        j = np.argmax(np.abs(u[:, 0]))
        basis_predictions[:, i] = u[:, 0] * np.sign(u[j, 0])
    
    # Use this if enough RAM
    # Shape (num_sources * num_windows,)
    loss_svd_basis = np.min(scipy.spatial.distance.cdist(basis_predictions.T, svd_basis.T), axis=1)
    return loss_svd_basis.mean()


def spike_metrics(pred_muaps: torch.Tensor, gt_muaps: torch.Tensor, tolerance: int = 5) -> tuple:
    """
    Compute the accuracy of the spike trains generated from MUAPs.
    
    Args:
        pred_muaps (torch.Tensor): Shape (n, m, T) - predicted MUAPs
        gt_muaps (torch.Tensor): Shape (n, m, T) - ground truth MUAPs
        tolerance (int): Time step tolerance for spike matching

    Returns:
        tuple: (accuracy, precision, recall, f1)
    """

    pred_spikes = muaps_to_spike_trains(pred_muaps)         # (n,) OR (b, n,)
    gt_spikes = muaps_to_spike_trains(gt_muaps)

    return calculate_spike_train_metrics(pred_spikes, gt_spikes, tolerance)


def muap_cross_correlation(muap1, muap2):
    """
    Calculate the maximum cross-correlation between two MUAPs across all possible shifts.
    
    Args:
        muap1, muap2 (torch.Tensor): MUAPs of shape (channels, time)
    Returns:
        float: Maximum correlation value
    """
    c1, t1 = muap1.shape
    c2, t2 = muap2.shape
    assert c1 == c2, "MUAPs must have the same number of channels"

    # Always slide the shorter over the longer
    if t2 > t1:
        muap1, muap2 = muap2, muap1
        t1, t2 = t2, t1

    max_corr = -float('inf')
    for shift in range(t1 - t2 + 1):
        window = muap1[:, shift:shift+t2]
        dot = (window * muap2).sum()
        norm = torch.norm(window) * torch.norm(muap2) + 1e-8
        corr = dot / norm
        if corr > max_corr:
            max_corr = corr.item()

    return max_corr


def muaps_and_spike_trains_long_window(test_muaps, test_spike_trains, all_pred_muaps, all_pred_spikes):
    """
    Calculate metrics for muaps and spike trains in a long window setting.
    It matches muaps using cross-correlation and then reorders the spike trains and computes their metrics.
    Only matched pairs are used for metric calculation.
    
    Args:
        test_muaps: Ground truth muaps
        test_spike_trains: Ground truth spike trains
        all_pred_muaps: Predicted muaps
        all_pred_spikes: Predicted spike trains
    
    Returns:
        tuple: (metrics_tuple, average_muap_similarity)
            - metrics_tuple: average of (accuracy, precision, recall, f1)
            - average_muap_similarity: Average correlation between matched pairs
    """
    # Convert inputs to torch tensors if they aren't already
    test_muaps = torch.as_tensor(test_muaps)
    test_spike_trains = torch.as_tensor(test_spike_trains)
    all_pred_muaps = torch.as_tensor(all_pred_muaps)
    all_pred_spikes = torch.as_tensor(all_pred_spikes)
    
    n_gt = len(test_muaps)
    n_pred = len(all_pred_muaps)
    
    # Create a correlation matrix between all pairs of MUAPs
    correlation_matrix = torch.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            correlation_matrix[i, j] = muap_cross_correlation(all_pred_muaps[i], test_muaps[j])
    
    # Match predicted MUAPs to ground truth MUAPs based on highest correlation
    matched_pairs = []  # List of (pred_idx, gt_idx) pairs
    matched_gt = set()
    matched_correlations = []
    
    # Sort all correlations in descending order
    correlations_flat = correlation_matrix.flatten()
    sorted_indices = torch.argsort(correlations_flat, descending=True)
    
    # Convert flat indices to 2D indices
    for flat_idx in sorted_indices:
        pred_idx = flat_idx // n_gt
        gt_idx = flat_idx % n_gt
        
        # If neither MUAP has been matched yet, create a match
        if pred_idx not in [p for p, _ in matched_pairs] and gt_idx not in matched_gt:
            matched_pairs.append((pred_idx.item(), gt_idx.item()))
            matched_gt.add(gt_idx.item())
            matched_correlations.append(correlation_matrix[pred_idx, gt_idx].item())
            
            # Stop if we've matched all ground truth MUAPs
            if len(matched_gt) == min(n_gt, n_pred):
                break

    average_muap_similarity = np.mean(matched_correlations)
    
    # Create tensors containing only the matched spike trains
    matched_pred_spikes = torch.stack([all_pred_spikes[pred_idx] for pred_idx, _ in matched_pairs])
    matched_gt_spikes = torch.stack([test_spike_trains[gt_idx] for _, gt_idx in matched_pairs])
    
    # Calculate metrics using only the matched pairs
    spike_train_metrics = calculate_spike_train_metrics(matched_pred_spikes, matched_gt_spikes)
    
    return spike_train_metrics, average_muap_similarity


def single_spike_train_metrics(pred_spikes, gt_spikes, tolerance):
    """
    Calculate metrics for single spike train.

    Args:
        pred_spikes: Predicted spike train of shape (T,)
        gt_spikes: Ground truth spike train of shape (T,)
    """
    tp, tn, fp, fn = spike_train_confusion_single(pred_spikes, gt_spikes, tolerance)

    # Calculate accuracy
    total = tp + tn + fp + fn
    if total == 0:
        accuracy = 0.0

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1-Score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    rate_of_aggreement = (2*tp) / ((2*tp) + fp + fn) if (tp + fp + fn) > 0 else 0

    return accuracy, precision, recall, f1, rate_of_aggreement


def calculate_spike_train_metrics(pred_spikes: torch.Tensor, gt_spikes: torch.Tensor, tolerance: int = 5) -> tuple:
    """Calculate metrics for spike trains.
    
    Args:
        pred_spikes (torch.Tensor): Predicted spike trains of shape (n, T) or (b, n, T)
        gt_spikes (torch.Tensor): Ground truth spike trains of shape (n, T) or (b, n, T)
        tolerance (int): Time step tolerance for spike matching

    Returns:
        tuple: (accuracy, precision, recall, f1, rate_of_agreement)
            - accuracy (float): Overall accuracy
            - precision (float): Precision score
            - recall (float): Recall score
            - f1 (float): F1 score
            - rate_of_agreement (float): Rate of agreement score
    """
    tp, tn, fp, fn = spike_train_confusion_counts(pred_spikes, gt_spikes, tolerance)

    # Calculate accuracy
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (total > 0) else 0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1-Score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    rate_of_aggreement = (2*tp) / ((2*tp) + fp + fn) if (tp + fp + fn) > 0 else 0

    return accuracy, precision, recall, f1, rate_of_aggreement


def muaps_to_spike_trains(muaps):
    """
    Convert MUAPs to spike trains by detecting the peak per source.
    Assumes one spike per source per window.
    
    Args:
        muaps (torch.Tensor): shape (b, n, m, T)
    
    Returns:
        torch.Tensor: binary spike trains of shape (b, n, T)
    """
    expand_batch = False
    if muaps.dim() == 3:
        muaps = muaps.unsqueeze(0)  # Add batch dimension if missing
        expand_batch = True

    b, n, m, T = muaps.shape
    
    # Compute the absolute sum across channels
    energy = muaps.abs().sum(dim=2)  # shape (b, n_sources, T)
    
    # Get the index of the max energy per source (most likely spike time)
    spike_indices = energy.argmax(dim=-1)  # shape (b, n_sources,)
    
    # Create binary spike trains
    spike_trains = torch.zeros_like(energy, dtype=torch.int16)
    batch_idx = torch.arange(b)[:, None]
    source_idx = torch.arange(n)[None, :]
    spike_trains[batch_idx, source_idx, spike_indices] = 1

    if expand_batch:
        spike_trains = spike_trains.squeeze(0)  # Remove batch dimension if it was added
    return spike_trains


def find_best_offset(pred_idx, gt_idx, max_offset=50, tolerance=5):
    """
    Find the best systematic offset between two spike trains.
    
    Args:
        pred_idx (torch.Tensor): Indices of predicted spikes
        gt_idx (torch.Tensor): Indices of ground truth spikes
        max_offset (int): Maximum offset to consider
        
    Returns:
        int: Best offset that maximizes the number of matches
    """
    best_matches = 0
    best_offset = 0
    
    # Convert max_offset to integer
    max_offset = int(max_offset)
    
    # Try different offsets
    for offset in range(-max_offset, max_offset + 1):
        # Shift predicted spikes by offset
        shifted_pred = pred_idx + offset
        
        # Count matches within tolerance
        matches = 0
        i, j = 0, 0
        while i < len(shifted_pred) and j < len(gt_idx):
            if abs(shifted_pred[i] - gt_idx[j]) <= tolerance: 
                matches += 1
                i += 1
                j += 1
            elif shifted_pred[i] < gt_idx[j]:
                i += 1
            else:
                j += 1
        
        if matches > best_matches:
            best_matches = matches
            best_offset = offset
    
    return best_offset


def spike_train_confusion_single(spike_pred, spike_true, tolerance):
    """
    Args: 
        spike_pred (np.ndarray | torch.Tensor): shape (T,)
        spike_true (np.ndarray | torch.Tensor): shape (T,)
    """
    T = spike_pred.shape[0]
    tp, tn, fp, fn = 0, 0, 0, 0

    if isinstance(spike_true, torch.Tensor):
        spike_true = spike_true.cpu().numpy()
    if isinstance(spike_pred, torch.Tensor):
        spike_pred = spike_pred.cpu().numpy()

    # Convert from spike trains to indexes of spikes
    # pred_idx = torch.nonzero(spike_pred).flatten()
    # gt_idx = torch.nonzero(spike_true).flatten()
    pred_idx = np.nonzero(spike_pred)[0]
    gt_idx = np.nonzero(spike_true)[0]
    
    # Find best systematic offset
    best_offset = find_best_offset(pred_idx, gt_idx, max_offset=50, tolerance=tolerance)
    
    # Apply the offset to predicted spikes
    pred_idx = pred_idx + best_offset
    
    # Create counters for the pred and gt respectively
    i = 0
    j = 0

    while i < len(pred_idx) and j < len(gt_idx):
        if abs(pred_idx[i] - gt_idx[j]) <= tolerance:
            i += 1
            j += 1
            tp += 1
        elif pred_idx[i] < gt_idx[j]:
            i += 1
            fp += 1
        else:
            j += 1
            fn += 1
    
    # Any remaining predictions must be false positives
    fp += len(pred_idx) - i
    # Any remaining ground truth spikes must be false negatives
    fn += len(gt_idx) - j
    
    tn = T - (tp + fp + fn)

    return tp, tn, fp, fn


def spike_train_confusion_counts(spike_pred, spike_true, tolerance):
    """
    Args: 
        spike_pred (np.ndarray | torch.Tensor): shape (n, T)
        spike_true (np.ndarray | torch.Tensor): shape (n, T)
    """
    expand_batch = False
    if len(spike_pred.shape) == 2:
        if isinstance(spike_pred, torch.Tensor):
            spike_pred = spike_pred.unsqueeze(0)  # Add batch dimension if missing
        else:
            spike_pred = np.expand_dims(spike_pred, axis=0)
        expand_batch = True
    if len(spike_true.shape) == 2:
        if isinstance(spike_true, torch.Tensor):
            spike_true = spike_true.unsqueeze(0)  # Add batch dimension if missing
        else:
            spike_true = np.expand_dims(spike_true, axis=0)

    B, n, T = spike_pred.shape
    tp, tn, fp, fn = 0, 0, 0, 0

    for b in range(B):
        for source_idx in range(n):
            tp_i, tn_i, fp_i, fn_i = spike_train_confusion_single(
                spike_pred[b, source_idx],
                spike_true[b, source_idx],
                tolerance
            )
            tp += tp_i
            tn += tn_i
            fp += fp_i
            fn += fn_i

    if expand_batch:
        spike_pred = spike_pred.squeeze(0)  # Remove batch dimension if it was added
        spike_true = spike_true.squeeze(0)

    return tp, tn, fp, fn


def generate_windows(data: np.ndarray, 
                     window_length: int, 
                     stride: int = 1):
    """
    Adapted from Kostia's code

    Generate overlapping windows from multi-channel time-series data.

    Args:
        data (np.ndarray): Shape (channels, time), e.g. (m, T)
        window_length (int): Length of each window (T)
        stride (int): Step size between consecutive windows

    Returns:
        np.ndarray: Shape (num_windows, channels, window_length)
    """
    m, T = data.shape
    n_windows = 1 + (T - window_length) // stride
    windows = np.zeros((n_windows, m, window_length), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_length
        windows[i] = data[:, start:end]

    return windows


def reconstruction_error(emg: np.ndarray, 
                         predicted_sources: np.ndarray, 
                         stride: int = 1,
                         num_windows_per_sample: int = 1) -> float:
    """
    Calculate the reconstruction error between the original EMG signal and the reconstructed signal from predicted sources.
    
    Code taken from Kostia

    Args:
        emg (np.ndarray): shape (m, T_total). tshe original EMG signal.
        predicted_sources (np.ndarray): shape (num_windows, n, m, T) Predicted sources from the model.
        stride (int): Stride for windowing the EMG signal.
        num_windows_per_sample (int): Number of windows per sample. This is used to ignore the first few windows that are not complete.
     
    Returns:
        float: Reconstruction loss.
    """
    _, n, m, T = predicted_sources.shape
    emg_windowed = generate_windows(emg, T, stride)
    # Ingnore first num_windows_per_sample - 1 windows
    # emg_windowed = emg_windowed[num_windows_per_sample - 1:]
    predictions_sum = np.sum(predicted_sources, axis=1)

    # Shape (num_windows, )
    loss_relative = np.linalg.norm(predictions_sum - emg_windowed, axis=(1, 2)) / np.linalg.norm(emg_windowed, axis=(1, 2))
    
    return loss_relative.mean()


def ica_score(emg: np.ndarray, predicted_sources: np.ndarray) -> float:
    """
    Compute ICA similarity score between predicted sources and ICA components
    from the original EMG signal.

    Args:
        emg (np.ndarray): shape (m, T_total). The original EMG signal.
        predicted_sources (np.ndarray): shape (num_windows, n, m, T).
            Predicted sources from the model.

    Returns:
        float: ICA score (average absolute correlation between predicted sources
               and ICA components).
    """

    return 1
