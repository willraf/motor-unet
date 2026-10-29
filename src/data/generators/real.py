"""
Module real

Provides a data generator for real EMG data.

Usage:
    from src.data.generators.real import RealDataGenerator
    generator = RealDataGenerator(cfg)
    emg, muaps, meta = generator()

Authors:
    Will Raftery

Date:
    04/09/2025
"""

import mat73
import numpy as np
import torch
import logging
from pathlib import Path

class RealDataGenerator:

    def __init__(self, cfg, emg=None, mode=None):
        self.cfg = cfg
        self.mode = mode

        if emg is None:
            self.emg = self._load_emg()
        else:
            self.emg = emg

        self.windows = self._window_emg(self.emg)
    
    def __len__(self):
        return len(self.windows) - self.cfg.data.num_windows + 1

    def __getitem__(self, idx):
        """
        Output shapes:
            emg: (W, m, T)
            muaps: None
            meta: {"window_idx": int}
        """
        num_windows = self.cfg.data.num_windows

        emg = self.windows[idx:idx + num_windows]  # (W, m, T)
        muaps = None
        meta = {"window_idx": idx, "signal_idx": idx*self.cfg.predict.stride}


        return emg, muaps, meta
    
    
    def _load_emg(self):
        """
        Load the data from .mat or .npy files.
        """
        data_path = self.cfg.predict.data_path
        data_path = Path(data_path)

        if data_path.suffix == '.mat':
            emg = self._load_emg_mat()
        elif data_path.suffix == '.npy':
            emg = self._load_emg_numpy()
        else:
            raise ValueError(f"Data file format {data_path.suffix} not recognised")

        return emg


    def _load_emg_mat(self):
        """
        Load the data from .mat files and extract the relevant sections.
        
        Args:
            data_dir (str): Path to the EMG matlab files.
            fname (str): Data filename
        
        Returns:
            np.ndarray: shape (num_channels, time_steps). EMG data.
        """
        data_path = self.cfg.predict.data_path
        data_path = Path(data_path)

        # Load the data
        data_dict = mat73.loadmat(data_path)
        data = data_dict['RecInfo']['DAQ']['EMGraw'][:, :-2].T
        # Get a 5000 step sample
        emg = data[:, 5000:10000]

        logging.info(f'Data loaded from {data_path}')

        return emg
    

    def _load_emg_numpy(self):
        """
        Load the data from .npy files and extract the relevant sections.
        
        Args:
            data_dir (str): Path to the EMG matlab files.
            fname (str): Data filename
        
        Returns:
            np.ndarray: shape (num_channels, time_steps). EMG data.
        """
        data_path = self.cfg.predict.data_path
        num_channels = self.cfg.data.num_channels
        data_path = Path(data_path)

        with data_path.open('rb') as f:
            emg_data = np.load(f)

        logging.info(f'Data loaded from {data_path} shape {emg_data.shape}')

        # if len(emg_data.shape) == 3:
        #     # Select third column
        #     emg_data = emg_data[2]

        # full_emg = emg_data[:, 35000:45000:2]
        
        # downsample_factor = full_emg.shape[0] // num_channels
        # downsampled_emg = full_emg[::downsample_factor, :]
        # downsampled_emg = downsampled_emg[:num_channels, :]
        channel_mask = [1, 5, 8, 11, 15, 18, 21, 24, 27, 30, 33, 36]
        downsampled_emg = emg_data[2, channel_mask, :]

        logging.info(f'Downsampled EMG shape: {downsampled_emg.shape}')

        return downsampled_emg
    

    def _window_emg(self, emg: np.ndarray) -> np.ndarray:
        window_length = self.cfg.data.window_length
        stride = self.cfg.predict.stride
    
        emg_tensor = torch.from_numpy(emg).float()
        windows = emg_tensor.unfold(dimension=1, size=window_length, step=stride)
        return windows.permute(1, 0, 2).numpy()  # (W, m, T)
