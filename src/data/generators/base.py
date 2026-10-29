from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Optional
from pathlib import Path
import numpy as np
import pickle
import logging

class BaseDataGenerator(ABC):
    """Base class for all synthetic data generators."""

    def __init__(self, cfg, mode) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        window_length = cfg.data.window_length

        if self.mode == 'train':
            muap_pkl_path = self.cfg.training.muap_pkl_path
            self.num_samples = cfg.training.num_samples
        elif self.mode == 'eval':
            muap_pkl_path = self.cfg.evaluation.muap_pkl_path
            self.num_samples = cfg.evaluation.num_samples
        elif self.mode == 'spike_train_eval':
            muap_pkl_path = self.cfg.spike_train_eval.muap_pkl_path
            self.num_samples = cfg.spike_train_eval.num_samples
        elif self.mode == 'spike_eval':
            muap_pkl_path = self.cfg.spike_eval.muap_pkl_path
            self.num_samples = cfg.spike_eval.num_samples
        elif self.mode == 'noura_eval':
            muap_pkl_path = self.cfg.noura_eval.muap_pkl_path
            self.num_samples = cfg.noura_eval.num_samples

        self.muap_dictionary = self.load_muaps(muap_pkl_path)  # List of (m, L) arrays
        # Filter out muaps that are too long
        self.muap_dictionary = [muap for muap in self.muap_dictionary if muap.shape[1] <= window_length]
        self.muap_dictionary = self._pad_muaps(self.muap_dictionary)  # Shape: (n, m, L)
        # Normalize all MUAPs to have max absolute amplitude of 1
        self.muap_dictionary = self._normalize_muaps(self.muap_dictionary)

    def __len__(self):
        return self.num_samples


    def load_muaps(self, muap_path: Union[Path, str]) -> List[np.ndarray]:
        """
        Load MUAPs from a pickle, npy, or mat file.
        """
        muap_path = Path(muap_path)
        assert muap_path.exists(), f"MUAP file not found: {muap_path}"

        if muap_path.suffix == '.pkl':
            with open(muap_path, 'rb') as f:
                muaps = pickle.load(f)
        elif muap_path.suffix == '.npy':
            muaps = np.load(muap_path, allow_pickle=True)
        elif muap_path.suffix == '.mat':
            from scipy.io import loadmat
            mat_data = loadmat(muap_path)
            # Assuming the variable name in the .mat file is 'muaps'
            if 'muaps' in mat_data:
                muaps = mat_data['muaps']
            else:
                raise ValueError(f"'muaps' variable not found in {muap_path}")
        else:
            raise ValueError(f"Unsupported file format: {muap_path.suffix}")

        logging.info(f"Loaded {len(muaps)} MUAPs from {muap_path}")
        return muaps

    def load_muaps(self, muap_pkl_path: Union[Path, str]) -> List[np.ndarray]:

        muap_pkl_path = Path(muap_pkl_path)
        assert muap_pkl_path.suffix == '.pkl', "MUAP file must be a .pkl"
        assert muap_pkl_path.exists(), "MUAP file not found"

        # Load muaps from pkl file
        with open(muap_pkl_path, 'rb') as f:
            muaps = pickle.load(f)

        logging.info(f"Loaded {len(muaps)} muaps from {muap_pkl_path}")
        return muaps

    def _normalize_muaps(self, muaps: np.ndarray) -> np.ndarray:
        """Normalize each MUAP to have maximum absolute amplitude of 1.
        
        Args:
            muaps (np.ndarray): Array of MUAPs with shape (n, m, L)
            
        Returns:
            np.ndarray: Normalized MUAPs with same shape
        """
        normalized_muaps = muaps.copy()
        
        for i in range(len(normalized_muaps)):
            max_abs_amplitude = np.max(np.abs(normalized_muaps[i]))
            if max_abs_amplitude > 0:  # Avoid division by zero
                normalized_muaps[i] = normalized_muaps[i] / max_abs_amplitude
                
        return normalized_muaps


    def _pad_muaps(self, muaps: List[np.ndarray]) -> np.ndarray:
        """Zero pad a list of MUAPs to be the same length.

        Args:
            muaps (List[np.ndarray]): List of muaps. Each muap is of shape (num_channels, length).

        Returns:
            np.ndarray: Padded MUAPs with shape (n, m, L) where:
                n: number of MUAPs
                m: number of channels
                L: maximum length of any MUAP
        """
        max_length = max(muap.shape[1] for muap in muaps)          # L
        padded_muaps = [np.pad(muap, ((0, 0), (0, max_length - muap.shape[1]))) for muap in muaps]
        return np.array(padded_muaps)  
    

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
            np.ndarray: Shape (n, m, T) List of muaps to use for generating the datapoint.
        """
        num_channels = self.cfg.data.num_channels

        if muaps is None:
            muaps = self.muap_dictionary  # Use the entire muap dictionary

        original_num_channels = muaps.shape[1]

        if channel_shift is None:
            channel_shift = np.random.randint(0, num_channels)

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


    def _shift_pad_muap(self, 
                        muap: np.ndarray, 
                        shift_amount: int = None
        ) -> np.ndarray:
        """
        Shift and pad a muap to the target length.

        Args:
            muap (np.ndarray): Shape (num_channels, length). The muaps to shift and pad. 
                Muaps from one source, as measured by each channel
            shift_amount (int|None): If provided, use this shift amount, else random.

        Returns:
            np.ndarray: The shifted and padded muap.
        """
        target_length = self.cfg.data.window_length
        allow_partial_fit = self.cfg.data.allow_partial_fit

        num_channels, muap_length = muap.shape
            
        if shift_amount is None:
            if allow_partial_fit:
                # Random shift so the MUAP fits partially
                shift_amount = np.random.randint(-muap_length + 1, target_length - 1)
            else:
                # Random shift so the MUAP fits entirely
                diff = target_length - muap_length
                if diff >= 0:
                    shift_amount = np.random.randint(0, diff + 1)  # include 0 to diff
                else:
                    shift_amount = np.random.randint(diff, 1)

        # Calculate how much of the MUAP fits in the window
        start_in_window = max(shift_amount, 0)
        end_in_window = min(shift_amount + muap_length, target_length)
        muap_start = start_in_window - shift_amount  # How much of the MUAP to use
        muap_end = end_in_window - shift_amount  # How much of the MUAP to use
        
        padded_muap = np.zeros((muap.shape[0], target_length))
        padded_muap[:, start_in_window:end_in_window] = muap[:, muap_start:muap_end]
        return padded_muap

                

    def _scale_muap_amplitudes(self, muaps: np.ndarray) -> np.ndarray:
        """Scale the amplitudes of the muap contributions

        Args:
            muap (np.ndarray): Shape (n, m, T). The contributions of each source to the EMG signal

        Returns:
            np.ndarray: Shape (n, m, T). The newly weighted muap contributions
        """
        num_sources = muaps.shape[0]
        amplitude_distribution = self.cfg.data.amplitude.distribution

        if amplitude_distribution == 'beta':         
            alpha_min = self.cfg.data.amplitude.alpha_min
            alpha_max = self.cfg.data.amplitude.alpha_max
            beta_min = self.cfg.data.amplitude.beta_min
            beta_max = self.cfg.data.amplitude.beta_max

            # Beta distribution: more small values, fewer large ones
            alpha = np.random.uniform(alpha_min, alpha_max)
            beta = np.random.uniform(beta_min, beta_max)

            amplitudes = np.random.beta(a=alpha, b=beta, size=num_sources)
        elif amplitude_distribution == 'kostia':
            amp_low = 0.5
            amplitudes = np.random.uniform(amp_low, 1, num_sources)
            amplitudes = np.sort(amplitudes)[::-1]
            amplitudes /= np.max(amplitudes)
        else:
            return muaps # No scaling, return as is
       
        # # Sort from largest to smallest amplitude MUAP
        # sorted_indices = np.argsort(amplitudes)[::-1]
        # amplitudes = amplitudes[sorted_indices]
        # muaps = muaps[sorted_indices]

        amplitudes = amplitudes[:, None, None]
        muaps *= amplitudes

        return muaps  # (n, m, T)


    def _pad_sources(self, sources: np.ndarray, max_sources: int) -> np.ndarray:
        """Pad the sources to have a consistent number of sources per window.
        
        Args:
            sources (np.ndarray): Shape (n, m, T). The muaps from each source
            max_sources(int): The number of sources to pad to.
        
        Returns:
            np.ndarray: Shape (max_sources_per_window, m, T). The padded sources.
        """
        n, m, T = sources.shape

        # Pad the sources to max_sources
        padded_sources = np.full((max_sources, m, T), np.nan, dtype=np.float32)
        padded_sources[:n] = sources
        return padded_sources

    
    def _normalise_window(self, emg: np.ndarray, sources: np.ndarray, muaps: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize EMG data to be between -1 and 1. Sources are normalised be the same factor to preserve relative contributions.

        Args:
            emg (np.ndarray): Shape (n, T). EMG data
            sources (np.ndarray): Shape (n, m, T). The muaps from each source
            (optional) muaps (np.ndarray): Shape (n, m, L). The muaps used for convolution

        Returns:
            Tuple[np.ndarray, np.ndarray]: Normalised emg and muaps
        """
        scale_factor = np.max(np.abs(emg))
        if scale_factor > 0:
            emg /= scale_factor
            sources /= scale_factor
            if muaps is not None:
                muaps /= scale_factor
                return emg, sources, muaps
            
        return emg, sources

