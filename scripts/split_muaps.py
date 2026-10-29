"""
Script split_muaps

Script to split muaps from a pickle file into training and testing sets. 

Usage:
    % python3 -m scripts.split_muaps

Authors:

Date:
    18/04/2025
"""

import pickle
import logging
import numpy as np
import argparse
import os
from omegaconf import OmegaConf

def load_muaps(muaps_path: str):
    """
    Load muaps from a pickle file.

    Args:
        muaps_path (str): Path to the muaps pickle file.

    Returns:
        List[np.ndarray]: List of muaps.
    """
    with open(muaps_path, 'rb') as f:
        muaps = pickle.load(f)
    logging.info(f"Loaded {len(muaps)} muaps from {muaps_path}")
    return muaps


def split(muaps: list, split_ratio: float) -> tuple:
    """
    Split muaps into training and testing sets.

    Args:
        muaps (list): List of muaps.
        split_ratio (float): Ratio for splitting the data. Default is 0.1.

    Returns:
        tuple: Training and testing muaps.
    """
    num_test_samples = int(len(muaps) * split_ratio)
    test_muaps = muaps[:num_test_samples]
    train_muaps = muaps[num_test_samples:]
    
    logging.info(f"Split {len(muaps)} muaps into {len(train_muaps)} training and {len(test_muaps)} testing samples.")
    
    return train_muaps, test_muaps


def save_muaps(muaps: list, muaps_path: str):
    """
    Save muaps to a pickle file.

    Args:
        muaps (list): List of muaps.
        muaps_path (str): Path to save the muaps.
    """
    with open(muaps_path, 'wb') as f:
        pickle.dump(muaps, f)
    logging.info(f"Saved {len(muaps)} muaps to {muaps_path}")


def get_global_stats(muaps):
    all_data = np.concatenate([muap.flatten() for muap in muaps])
    mean = np.mean(all_data)
    var = np.var(all_data)
    return mean, var


def get_per_channel_stats(muaps):
    num_channels = muaps[0].shape[0]
    all_data = np.concatenate([muap for muap in muaps], axis=1)  # shape: (channels, total_length)
    mean = np.mean(all_data, axis=1)
    var = np.var(all_data, axis=1)
    return mean, var


def main(muaps_path, train_muaps_path, test_muaps_path, test_split=0.1):
    """
    Main function to load, split, and save muaps.
    """

    # Load muaps
    muaps = load_muaps(muaps_path)

    # Shuffle muaps
    np.random.shuffle(muaps)

    mean, var = get_global_stats(muaps)

    # Split muaps into training and testing sets
    train_muaps, test_muaps = split(muaps, test_split)

    # Save the split muaps
    save_muaps(train_muaps, train_muaps_path)
    save_muaps(test_muaps, test_muaps_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split muaps into training and testing sets")

    parser.add_argument(
        '--muap_path',
        type=str,
        default="data/raw/simulated/muaps_48ch_subj1.pkl",
        help="Path to the muap pickle file"
    )
    parser.add_argument(
        '--testsplit',
        type=float,
        default=0.1,
        help="Fraction of muaps to use for testing"
    )

    parser.add_argument(
        '--train_path',
        type=str,
        default="data/raw/simulated/train_muaps_48ch_subj1.pkl",
        help="Path to save training muap split"
    )

    parser.add_argument(
        '--test_path',
        type=str,
        default="data/raw/simulated/test_muaps_48ch_subj1.pkl",
        help="Path to save testing muap split"
    )

    # Parse the arguments
    args = parser.parse_args()

    muaps_path = args.muap_path
    train_muaps_path = args.train_path
    test_muaps_path = args.test_path
    test_split = args.testsplit

    main(muaps_path, test_split, train_muaps_path, test_muaps_path)
