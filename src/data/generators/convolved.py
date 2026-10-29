"""Module for generating synthetic sEMG data.

This module contains the DataGenerator class which generates synthetic sEMG data by
convolving simulated spike trains with MUAPs (Motor Unit Action Potentials).

Usage:
    >>> from src.data.data_generator import DataGenerator
    >>> gen = DataGenerator(muaps, length=64, min_firing_rate=5, max_firing_rate=30)
    >>> emg, muap_contributions = gen.generate_datapoint(num_sources=5)

Authors:

Date:
    04/09/2025

Note:
    - There will be no/few edge cases at the start of window due to maup lag
    - Time is the LAST dimension in matrices, unlike ML standard (first dim)
    - Firing rates are randomly generated between min_firing_rate and max_firing_rate Hz
    - If a spike is generated near the end of the window, that source may be all zero after convolution
"""
import numpy as np
from typing import Tuple, Optional
import torch

from src.data.generators.base import BaseDataGenerator

class ConvolvedDataGenerator(BaseDataGenerator):

    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

    def __getitem__(self, idx):
        """
        Output shapes:
            emg: (W, m, T)
            muaps: (W, n, m, T)
            meta: {"spikes": (W, n, T)}
        """
        emg, muap_contributions, meta = self._generate_datapoint(return_spikes=True)

        if 'spike_trains' in meta:
            emg, muap_contributions, meta['spike_trains'] = self._window_data(emg, muap_contributions, meta.pop('spike_trains'))
        else:
            emg, muap_contributions = self._window_data(emg, muap_contributions)
        
        return emg, muap_contributions, meta  # (W, m, T), (W, n, m, T), {"spikes": (W, n, T), "muaps": (n, m, L)}


    def _window_data(self, emg: np.ndarray, sources: np.ndarray, spikes: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply windowing to the EMG and sources tensors.

        Args:
            emg (np.ndarray): Shape (m, T). The EMG signal
            sources (np.ndarray): Shape (n, m, T). The source contributions
            spikes (Optional[np.ndarray]): Shape (n, T). The spike trains

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of:
                - emg_windows: Shape (W, m, T)
                - source_windows: Shape (W, n, m, T)
        """
        window_length = self.cfg.data.window_length
        stride = self.cfg.data.stride


        emg = torch.from_numpy(emg).float()
        sources = torch.from_numpy(sources).float()


        # Unfold over time dimension
        emg_windows = emg.unfold(dimension=1, size=window_length, step=stride)  # (m, num_windows, T)
        sources_windows = sources.unfold(dimension=2, size=window_length, step=stride)    # (n, m, num_windows, T)

        emg_windows = emg_windows.permute(1, 0, 2)          # (num_windows, m, T)
        sources_windows = sources_windows.permute(2, 0, 1, 3)         # (num_windows, n, m, T)

        if spikes is not None:
            spikes = torch.from_numpy(spikes).float()
            spikes_windows = spikes.unfold(dimension=1, size=window_length, step=stride)  # (n, num_windows, T)
            spikes_windows = spikes_windows.permute(1, 0, 2)  # (num_windows, n, T)
            return emg_windows, sources_windows, spikes_windows
        

        return emg_windows, sources_windows


    def _generate_spike_times_poisson(self, firing_rates: np.ndarray) -> np.ndarray:
        """Simulate spike times from multiple sources using a Poisson process.

        Args:   
            firing_rates (np.ndarray): Shape (n,) Firing rates of each source/MU.

        Returns:
            np.ndarray: Shape (s, n). Time stamps of spikes (MUAPs)
                - s: Max number of spikes
                - n: Number of sources
        """
        duration = self.cfg.data.duration

        num_sources = firing_rates.shape[0]
        max_spikes = int(duration * np.max(firing_rates) * 1.5)
        
        # Generate inter-spike intervals using exponential distribution
        isi = np.random.exponential(1 / firing_rates[:, None], size=(num_sources, max_spikes))
        spike_times = np.cumsum(isi, axis=1)
        
        # Mask spikes beyond simulation duration
        spike_times[spike_times >= duration] = np.nan
        spike_times = spike_times.T  # Shape: (s, n)
        
        # Handle case of no spikes
        if spike_times.shape[0] == 0:
            return np.full((1, num_sources), np.nan)
        
        return spike_times


    def _generate_spike_times_gamma(self, firing_rates: np.ndarray) -> np.ndarray:
        """Simulate spike times from multiple sources using a Gamma process.

        Args:
            firing_rates (np.ndarray): Shape (n,) Firing rates of each source/MU.

        Returns:
            np.ndarray: Shape (s, n). Time stamps of spikes (MUAPs)
                - s: Max number of spikes
                - n: Number of sources
        """
        duration = self.cfg.data.duration
        shape = self.cfg.data.spikes.shape

        num_sources = firing_rates.shape[0]
        max_spikes = int(duration * np.max(firing_rates) * 1.5)

        # Scale parameter for gamma distribution: θ = 1 / (rate * shape)
        scale = 1 / (firing_rates * shape)[:, None]  # Shape: (n, 1)

        # Draw ISIs from gamma distribution
        isi = np.random.gamma(shape=shape, scale=scale, size=(num_sources, max_spikes))

        # Cumulative sum to get spike times
        spike_times = np.cumsum(isi, axis=1)  # Shape: (n, s)

        # Mask spikes beyond simulation duration
        spike_times[spike_times >= duration] = np.nan

        # Transpose to match expected shape: (s, n)
        spike_times = spike_times.T

        # Handle case of no spikes
        if spike_times.shape[0] == 0:
            return np.full((1, num_sources), np.nan)

        return spike_times


    def _spike_trains_to_times(self, spike_trains: np.ndarray) -> np.ndarray:
        """Convert spike trains (delta functions) to spike times.

        Args:
            spike_trains (np.ndarray): Shape (n, t). Discrete binary matrix of spike trains.
                - n: Number of sources
                - t: Number of time steps

        Returns:
            np.ndarray: Shape (s, n). Time stamps of spikes (MUAPs)
                - s: Max number of spikes per source
                - n: Number of sources
        """
        sampling_rate = self.cfg.data.sampling_rate

        num_sources = spike_trains.shape[0]
        spike_indexes = [np.where(spike_trains[source] == 1)[0] for source in range(num_sources)]
        spike_times = [indexes * sampling_rate for indexes in spike_indexes]
        
        max_spikes = max(len(times) for times in spike_times)
        
        # Pad shorter arrays with NaN to create a uniform shape
        spike_times_padded = np.full((max_spikes, num_sources), np.nan)
        
        for source in range(num_sources):
            spike_times_padded[:len(spike_times[source]), source] = spike_times[source]
        
        return spike_times_padded


    def _spike_times_to_trains(self, spike_times: np.ndarray) -> np.ndarray:
        """Convert spike times to spike trains (delta functions).

        Args: 
            spike_times (np.ndarray): Shape (s, n). Time stamps of spikes (MUAPs)
                - s: Max number of spikes
                - n: Number of sources

        Returns:
            np.ndarray: Shape (n, t). Binary matrix where:
                n: number of sources
                t: number of time steps
        """
        sampling_period = self.cfg.data.sampling_period
        length = self.cfg.data.length

        _, num_sources = spike_times.shape

        # Convert None values of spike times (where simulation for that source has ended)
        # into -1 to prevent type conversion error 
        spike_times = np.where(np.isnan(spike_times), -1, spike_times)
        spike_indexes = (spike_times // sampling_period).astype(int)
        
        spike_matrix = np.zeros((num_sources, length), dtype=int)

        # Filter out invalid indexes (e.g., -1 from NaN values)
        for source in range(num_sources):
            valid_indexes = spike_indexes[:, source]
            valid_indexes = valid_indexes[valid_indexes >= 0]
            spike_matrix[source, valid_indexes] = 1
            
        return spike_matrix


    def _lagged_spike_trains(self, spike_trains: np.ndarray, L: int) -> np.ndarray:
        """Transform spike trains into lagged spike trains for vectorized convolution.

        Args:
            spike_trains (np.ndarray): Shape (n, T). Tensor of spike trains
            L (int): Length of the filter which will convolve with the spike trains

        Returns:
            np.ndarray: Shape (n, L, T). Tensor of lagged spike trains. 
                S_{ijk} is impulse (0 or 1) of source i at time (k - j)
        """
        n, T = spike_trains.shape
        t_indices = np.arange(T)
        l_indices = np.arange(L)

        # Compute the shifted indices (broadcasting L over T)
        # Matrix of indeces with X_ij = j - i
        shifted_indices = t_indices[None, :] - l_indices[:, None]
        valid_mask = shifted_indices >= 0
        
        S = np.zeros((n, L, T))
        for j in range(n):
            S[j] = np.where(valid_mask, spike_trains[j, shifted_indices], 0)
            
        return S


    def _generate_spike_trains(self, num_sources: int) -> np.ndarray:
        """generate spike trains for the specified number of sources.

        Generates random firing rates for each source and simulates spike times.
        Args:
            num_sources (int): Number of sources to simulate

        Returns:
            np.ndarray: Shape (n, T). The spike trains of each source as binary matrix
                - n: Number of sources
                - T: Length of the signal
        """
        min_firing_rate = self.cfg.data.spikes.min_firing_rate
        max_firing_rate = self.cfg.data.spikes.max_firing_rate
        spike_distribution = self.cfg.data.spikes.distribution
        duration = self.cfg.data.duration

        firing_rates = np.random.randint(min_firing_rate, max_firing_rate, (num_sources))
        if spike_distribution == 'poisson':
            spike_times = self._generate_spike_times_poisson(firing_rates)
        elif spike_distribution == 'gamma':
            spike_times = self._generate_spike_times_gamma(firing_rates)  # (s, n)
        else:
            raise ValueError(f"Unknown spike distribution: {spike_distribution}")
        # Ensure at least one spike per source
        mask = np.all(np.isnan(spike_times), axis=0)
        spike_times[0, mask] = np.random.uniform(low=0, high=duration, size=(1, np.sum(mask)))  # (1, n)
            
        return self._spike_times_to_trains(spike_times=spike_times)        # (n, T)


    def _generate_datapoint(self, return_spikes: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single datapoint of synthetic sEMG data.

        Args:
            num_sources (Optional[int]): Number of sources to simulate. If None, randomly generated between min_sources and max_sources.
            pad_sources (Optional[bool]): Whether to pad sources to max_sources.
            return_spikes (Optional[bool]): Whether to return spike trains and muaps instead of muap contributions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (emg, muap_contributions) where:
                emg: Array of shape (m, T) containing simulated sEMG signals
                muap_contributions: Array of shape (n, m, T) containing source contributions

            OR

            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tupele of (emg, spike_trains, muaps) where:
                emg: Shape (n, m, T)
                spike_trains: Shape (n, T) - Binary matrix of spike trains
                muaps: Shape (n, m, L) - The MUAPs used for convolution
        """
        min_sources = self.cfg.data.min_sources
        max_sources = self.cfg.data.max_sources
        normalise = self.cfg.data.normalise
        add_noise = self.cfg.data.noise.add
        min_noise_std = self.cfg.data.noise.min_std
        max_noise_std = self.cfg.data.noise.max_std
        num_channels = self.cfg.data.num_channels

        pad_sources = True    # Could move to arguments for more flexibility

        num_sources = np.random.randint(min_sources, max_sources+1)

        spike_trains = self._generate_spike_trains(num_sources)  # (n, T)
        
        muaps = self._get_muaps(num_sources)    # (n, m, L) 
        # Vary the amplites of the muaps 
        muaps = self._scale_muap_amplitudes(muaps) 

        # Convolve muaps with spike trains to get their contribution to the EMG signal
        muap_contributions = self._convolve(spike_trains, muaps)  # (n, m, T)

        # Check for zero sources and regenerate if needed
        zero_sources = np.all(np.abs(muap_contributions) < 1e-6, axis=(1, 2))
        while np.any(zero_sources):
            # Regenerate only the zero sources
            zero_indices = np.where(zero_sources)[0]
            
            spike_trains[zero_indices] = self._generate_spike_trains(len(zero_indices))  # (n, T)
            muaps[zero_indices] = self._get_muaps(len(zero_indices))  # (n, m, L)
            muap_contributions[zero_indices] = self._convolve(spike_trains[zero_indices], muaps[zero_indices])  # (n, m, T)
       
            # Check again for zero sources
            zero_sources = np.all(np.abs(muap_contributions) < 1e-6, axis=(1, 2))

        emg = np.sum(muap_contributions, axis=0)  # (m, T)
        
        if normalise:
            emg, muap_contributions, muaps = self._normalise_window(emg, muap_contributions, muaps)

        # Add noise to the EMG signal
        if add_noise:
            # Random noise  level for each channel
            noise_stds = np.random.uniform(min_noise_std, max_noise_std, size=(num_channels, 1))
            noise = np.random.normal(scale=noise_stds, size=emg.shape).astype(np.float32)
            emg += noise

        if pad_sources:
            n, length = spike_trains.shape
            padded_spikes = np.full((max_sources, length), np.nan, dtype=np.float32)
            padded_spikes[:num_sources] = spike_trains
            spike_trains = padded_spikes

            muap_contributions = self._pad_sources(muap_contributions, max_sources)

        if return_spikes:

            return emg, muap_contributions, {'spike_trains': spike_trains, 'muaps': muaps}  # (m, T), (n, T), (n, m, L)

        return emg, muap_contributions, {}


    def _convolve(self, spike_trains: np.ndarray, muaps: np.ndarray) -> np.ndarray:
        """Convolve spike trains with MUAPs to get source contributions.

        Args:
            spike_trains (np.ndarray): Shape (n, T). The spike train of each source
            muaps (np.ndarray): Shape (n, m, L). The MUAPs of each source

        Returns:
            np.ndarray: Shape (n, m, T). Source contributions to the EMG signal
        """
        assert spike_trains.shape[0] == muaps.shape[0], "Number of sources in spike trains and muaps must match"
        
        _, T = spike_trains.shape
        n, m, L = muaps.shape
        S_prime = self._lagged_spike_trains(spike_trains=spike_trains, L=L)  # (n, L, T)

        #Equivalent to: for i in n:  muap_contributions[i] = muaps[i] @ S_prime[i]  # (m, L) @ (L, T) = (m, T)
        muap_contributions = np.einsum('nml,nlt->nmt', muaps, S_prime)
        return muap_contributions