from omegaconf import OmegaConf
from typing import Callable, List
import torch

import src.utils.visualisation as vis

from src.data import BaseDataset

import time
import torch

class GeneralDataset(BaseDataset):
    """
    General purpose dataset for handling synthetic or real data generated from a generator.
    """
    def __init__(self, 
                 cfg: OmegaConf, 
                 generator: Callable, 
                 transforms: List[Callable], 
                 collator: Callable, 
        ):
        super().__init__(cfg)
        self.generator = generator
        self.transforms = transforms
        self.collator = collator

    def __getitem__(self, idx):

        emg, muaps, meta = self.generator[idx]

        emg = torch.tensor(emg, dtype=torch.float32)
        if muaps is not None:
            muaps = torch.tensor(muaps, dtype=torch.float32)

        for t in self.transforms:
            emg, muaps, meta = t(emg, muaps, meta)

        emg, muaps, meta = self.collator(emg, muaps, meta)

        if muaps is None:
            muaps = torch.empty(0)

        return emg, muaps, meta

    def __len__(self):
        return len(self.generator)
