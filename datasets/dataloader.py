import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from collections import deque
import itertools

class ProteinLigandDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        self.target_batch_size = batch_size
        self.sample_buffer = deque()
        
        super().__init__(
            dataset,
            batch_size=batch_size * 2,  # Request more samples to handle Nones
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            drop_last=False,  # Changed to False to get all samples
            **kwargs
        )

    def collate_fn(self, batch):
        # Add valid samples to buffer
        valid_data = [data for data in batch if data is not None]
        self.sample_buffer.extend(valid_data)
        
        # If we have enough samples, return a full batch
        if len(self.sample_buffer) >= self.target_batch_size:
            # Take exactly batch_size samples
            batch_data = [self.sample_buffer.popleft() for _ in range(self.target_batch_size)]
            return Batch.from_data_list(batch_data)
            
        # Not enough samples yet
        return None

    def __iter__(self):
        self.sample_buffer.clear()  # Clear buffer at start of iteration
        
        # Create an infinite iterator over the dataset
        infinite_iterator = itertools.cycle(super().__iter__())
        
        while True:
            batch = next(infinite_iterator)
            if batch is not None:
                yield batch