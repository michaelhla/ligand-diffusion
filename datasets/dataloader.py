import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from collections import deque
import itertools

class ProteinLigandDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, interface_cutoff=8.0, smiles_tokenizer=None, **kwargs):
        """
        Args:
            dataset: Dataset containing protein-ligand pairs
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset
            num_workers: Number of workers for data loading
            interface_cutoff: Distance cutoff for interface residues
            smiles_tokenizer: Instance of SMILESBPETokenizer
        """
        self.smiles_tokenizer = smiles_tokenizer
        self.interface_cutoff = interface_cutoff
        self.sample_buffer = deque()
        self.target_batch_size = batch_size
        
        super().__init__(
            dataset,
            batch_size=batch_size * 2,  # Request more samples to handle Nones
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            drop_last=False,
            **kwargs
        )

    def get_interface_residues(
        self,
        protein_coords: torch.Tensor,  # Shape: (num_residues, 3, 3)
        ligand_coords: torch.Tensor,   # Shape: (num_ligand_atoms, 3)
        residue_indices: torch.Tensor  # Shape: (num_residues,)
    ) -> torch.Tensor:
        """
        Calculate interface residues based on distance to ligand atoms.
        Now handles backbone-only protein coordinates.
        """
        # Use CA atoms (second atom of each residue) for distance calculation
        ca_coords = protein_coords[:, 1, :]  # Shape: (num_residues, 3)
        
        # Reshape for broadcasting
        protein_coords_expanded = ca_coords.unsqueeze(1)  # (num_residues, 1, 3)
        ligand_coords_expanded = ligand_coords.unsqueeze(0)   # (1, num_ligand_atoms, 3)
        
        # Calculate pairwise distances between CA atoms and ligand atoms
        distances = torch.sqrt(((protein_coords_expanded - ligand_coords_expanded) ** 2).sum(dim=2))  # (num_residues, num_ligand_atoms)
        
        # Get minimum distance for each residue
        min_distances = distances.min(dim=1)[0]  # (num_residues,)
        
        # Create interface mask (True for residues within cutoff distance)
        interface_mask = min_distances < self.interface_cutoff
        
        return interface_mask

    def collate_fn(self, batch):
        """
        Collate function that only pads what's needed for the model.
        """
        # Remove None samples
        valid_data = [data for data in batch if data is not None]
        
        if not valid_data:
            return None

        # Find max lengths in this batch
        max_protein_length = max(data['protein'].pos.size(0) for data in valid_data)
        max_smiles_length = max(len(self.smiles_tokenizer.encode(data['ligand'].smiles)) for data in valid_data)
        
        # Lists to store batch data
        protein_data_list = []
        smiles_tokens_list = []
        
        for data in valid_data:
            # Get protein length
            protein_length = data['protein'].pos.size(0)
            
            # Pad protein coordinates
            padded_coords = torch.zeros((max_protein_length, 3, 3), dtype=torch.float32)
            padded_coords[:protein_length] = data['protein'].pos
            
            # Pad residue indices
            padded_residue_indices = torch.zeros(max_protein_length, dtype=torch.long)
            padded_residue_indices[:protein_length] = data['protein'].residue_indices
            
            # Store protein data
            protein_data_list.append({
                'coords': padded_coords,
                'residue_indices': padded_residue_indices
            })
            
            # Tokenize and pad SMILES
            smiles_tokens = torch.tensor(self.smiles_tokenizer.encode(data['ligand'].smiles), dtype=torch.long)
            padded_smiles = torch.zeros(max_smiles_length, dtype=torch.long)
            padded_smiles[:len(smiles_tokens)] = smiles_tokens
            smiles_tokens_list.append(padded_smiles)
        
        return {
            'protein': {
                'coords': torch.stack([d['coords'] for d in protein_data_list]),
                'residue_indices': torch.stack([d['residue_indices'] for d in protein_data_list])
            },
            'ligand': {
                'smiles_tokens': torch.stack(smiles_tokens_list)
            }
        }

    def __iter__(self):
        self.sample_buffer.clear()  # Clear buffer at start of iteration
        
        # Create an infinite iterator over the dataset
        infinite_iterator = itertools.cycle(super().__iter__())
        
        while True:
            batch = next(infinite_iterator)
            if batch is not None:
                yield batch