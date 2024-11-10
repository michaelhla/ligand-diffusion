import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

class ProteinLigandDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            **kwargs
        )

    @staticmethod
    def collate_fn(batch):
        # Filter out None values
        batch = [data for data in batch if data is not None]
        return Batch.from_data_list(batch)
    

