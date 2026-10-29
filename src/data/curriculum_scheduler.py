""" Module curriculum_scheduler

Implements a curriculum scheduler that updates the difficulty of a dataset based on specified milestones.

Usage:
    from src.data.curriculum_scheduler import CurriculumScheduler

    scheduler = CurriculumScheduler(
        dataset=my_dataset,
        milestones=[5, 10, 15],
        parameters=[{'difficulty': 'easy'}, {'difficulty': 'medium'}, {'difficulty': 'hard'}]
    )

    for epoch in range(num_epochs):
        scheduler.update(epoch)

Authors:
    Will Raftery

Date:
    31/07/2025

"""


from typing import List
import logging
from src.data.base_dataset import BaseDataset

class CurriculumScheduler:
    def __init__(self, 
                 dataset: BaseDataset,
                 milestones: List[int],
                 parameters: List[dict]):
        """
        Args:
            dataset: the dataset whose difficulty will be updated.
            milestones: epochs at which difficulty should change.
            parameters: a list of dicts with kwargs to pass to `dataset.set_difficulty`.
        """
        self.dataset = dataset
        self.milestones = milestones
        self.parameters = parameters

    def update(self, epoch: int):
        for milestone, params in zip(self.milestones, self.parameters):
            if epoch == milestone:
                logging.info(f"[Curriculum] Updating difficulty at epoch {epoch}")
                self.dataset.update_parameters(**params)