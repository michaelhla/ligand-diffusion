import torch
from torch.utils.data import DataLoader, ConcatDataset
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
    

from datasets.pdbbind import PDBBind
from datasets import MOAD
from datasets.dataloader import ProteinLigandDataLoader

# Create individual datasets
pdbbind_dataset = PDBBind(
    root='path/to/pdbbind',
    cache_path='data/cache',
    split='train',
    num_workers=4
)

moad_dataset = MOAD(
    root='path/to/moad',
    cache_path='data/cache',
    split='train',
    num_workers=4
)

# Combine datasets
combined_dataset = ConcatDataset([pdbbind_dataset, moad_dataset])

# Create dataloader with combined dataset
train_loader = ProteinLigandDataLoader(
    combined_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Use in training
for batch in train_loader:
    # batch.protein.pos - protein coordinates
    # batch.protein.residues - protein residues
    # batch.ligand.pos - ligand coordinates
    # batch.ligand.atom_types - ligand atom types
    # batch.ligand.smiles - ligand SMILES strings
    pass