"""Module for generating synthetic sEMG data.

This module contains the DataGenerator class which generates synthetic sEMG data by
by simply shifting, padding and summing muaps.
"""
import numpy as np
from typing import List, Optional, Tuple


from src.data.generators.base import BaseDataGenerator
import src.utils.visualisation as vis

class SingleSpikeDataGenerator(BaseDataGenerator):

    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)


    def __getitem__(self, idx):
        """
        Output shapes:
            emg: (W, m, T)
            muaps: (W, n, m, T)
            meta: {}
        """

        emg, muaps = self._generate_windows()
        meta = {}
        return emg, muaps, meta  # (W, m, T), (W, n, m, T), {}


    def _generate_windows(self):
        
        min_sources = self.cfg.data.min_sources
        max_sources = self.cfg.data.max_sources
        window_length = self.cfg.data.window_length
        num_windows = self.cfg.data.num_windows
        min_sources_per_window = self.cfg.data.min_sources_per_window
        max_sources_per_window = self.cfg.data.max_sources_per_window
        num_channels = self.cfg.data.num_channels

        # Ensure the sources per window arent out of the range of the total number of sources
        max_sources = min(max_sources, len(self.muap_dictionary))
        max_sources_per_window = min(max_sources_per_window, max_sources)

        total_num_sources = np.random.randint(min_sources, max_sources + 1)

        # Randomly select MUAPs without replacement
        indices = np.random.choice(len(self.muap_dictionary), size=total_num_sources, replace=False)
        muaps = self.muap_dictionary[indices]  # (n, m, L)

        # Vary the amplites of the muaps 
        muaps = self._scale_muap_amplitudes(muaps)

        num_sources_per_window = np.random.randint(min_sources_per_window, max_sources_per_window + 1, size=num_windows).tolist()

        # Choose 'starting channel' for rotation augmentation
        channel_shift = np.random.randint(0, num_channels)
              
        emg_windows = []
        sources_windows = []

        for i in num_sources_per_window:
            emg, sources = self._generate_window(
                num_sources=i, 
                muaps=muaps,
                channel_shift=channel_shift,    # want the same shift for all windows  
            )
            sources = self._pad_sources(sources, max_sources_per_window)
            emg_windows.append(emg)
            sources_windows.append(sources)
            
        emg_windows = np.stack(emg_windows, axis=0)                                 # W, m, T
        sources_windows = np.stack(sources_windows, axis=0)                         # W, n, m, T

        return emg_windows, sources_windows  # (W, m, T), (W, n, m, T)


    def _generate_window(self, num_sources: int, muaps: np.ndarray, channel_shift: int) -> Tuple[np.ndarray, np.ndarray]:

        normalise = self.cfg.data.normalise
        add_noise = self.cfg.data.noise.add
        min_noise_std = self.cfg.data.noise.min_std
        max_noise_std = self.cfg.data.noise.max_std
        num_channels = self.cfg.data.num_channels

        muaps_of_interest = self._get_muaps(num_sources, muaps=muaps, channel_shift=channel_shift)  # (n, m, T)
        emg = np.sum(muaps_of_interest, axis=0)  # (m, T)

        if normalise:
            emg, muaps_of_interest = self._normalise_window(emg, muaps_of_interest)

        # Add noise   to the EMG signal
        if add_noise:
            # Random noise  level for each channel
            noise_stds = np.random.uniform(min_noise_std, max_noise_std, size=(num_channels, 1))
            noise = np.random.normal(scale=noise_stds, size=emg.shape).astype(np.float32)
            emg += noise

        return emg, muaps_of_interest  # (m, T), (n, m, T)
    

    def _get_muaps(self, 
                   num_sources,
                   muaps: Optional[np.ndarray] = None, 
                   channel_shift: int = None) -> np.ndarray:
        """Get and process number of muaps from the muap dictionary or provided list of muaps.

        Args:
            num_sources (int): Number of muaps to get.
            muaps (Optional[np.ndarray]): Optional pre-defined muaps to use instead of entire muap dictionary.
            channel_shift (Optional[int]): Optional channel shift for the muaps.
        
        Returns:
            np.ndarray: Shape (num_sources, m, T) List of muaps to use for generating the datapoint.
        """
        if muaps is None:
            muaps = self.muap_dictionary  # Use the entire muap dictionary

        num_channels = self.cfg.data.num_channels
        original_num_channels = muaps.shape[1]

        indices = np.random.choice(len(muaps), size=num_sources, replace=False)
        muaps_of_interest = muaps[indices]  # (n, m, L)

        processed_muaps = []
        for muap in muaps_of_interest:
            step = original_num_channels // num_channels
            muap = np.roll(muap, shift=channel_shift, axis=0)[::step]  # Shift channels
            muap = muap[:num_channels]  # Downsample to output_num_channels in case self.num_channels is not divisible by self.output_num_channels

            muap = self._shift_pad_muap(muap)
            processed_muaps.append(muap)

        return np.stack(processed_muaps, axis=0)  # Shape (n, m, T)
    

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from src.utils.utils import load_config, set_seed

    parser = argparse.ArgumentParser(description="Test SingleSpikeDataGenerator")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    random_seed = cfg.experiment.random_seed
    set_seed(random_seed)

    data_generator = SingleSpikeDataGenerator(cfg, mode='train')

    emg, muaps, meta = data_generator[0]

    print(f"EMG shape: {emg.shape}")          # Expected: (m, T)
    print(f"MUAPs shape: {muaps.shape}")      # Expected: (n, m, T)

    vis.plot_emg_muaps_prediction(emg[0], muaps[0], muaps[0], fpath=None)

    print(f"EMG shape: {emg.shape}")          # Expected: (m, T)
    print(f"MUAPs shape: {muaps.shape}")      # Expected: (n, m, T)