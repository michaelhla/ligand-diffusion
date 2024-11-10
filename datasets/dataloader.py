import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from collections import deque
import itertools

class ProteinLigandDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        # Create vocabulary for residues and atoms
        self.residue_vocab = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'UNK': 20  # Unknown/other residues
        }
        
        self.atom_vocab = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 
            'P': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'H': 9,
            'UNK': 10  # Unknown atoms
        }
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
        
        # Tokenize residues and atoms for each sample
        for data in valid_data:
            # Convert residues to token indices
            residue_tokens = torch.tensor([
                self.residue_vocab.get(res, self.residue_vocab['UNK']) 
                for res in data['protein'].residues
            ], dtype=torch.long)
            data['protein'].residue_tokens = residue_tokens
            
            # Convert atom types to token indices
            atom_tokens = torch.tensor([
                self.atom_vocab.get(atom, self.atom_vocab['UNK'])
                for atom in data['ligand'].atom_types
            ], dtype=torch.long)
            data['ligand'].atom_tokens = atom_tokens
            
        self.sample_buffer.extend(valid_data)

    def __iter__(self):
        self.sample_buffer.clear()  # Clear buffer at start of iteration
        
        # Create an infinite iterator over the dataset
        infinite_iterator = itertools.cycle(super().__iter__())
        
        while True:
            batch = next(infinite_iterator)
            if batch is not None:
                yield batch