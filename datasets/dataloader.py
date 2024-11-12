import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from collections import deque
import itertools

class ProteinLigandDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, interface_cutoff=8.0, smiles_tokenizer=None, **kwargs):
        # Create vocabulary for residues and atoms
        self.smiles_tokenizer = smiles_tokenizer
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
            'UNK': 10,  # Unknown atoms
            # Special tokens for SMILES parsing
            'PAD': 11,  # Padding token
            'START': 12,  # Start of sequence
            'END': 13,  # End of sequence   
            '(': 14, ')': 15,  # Brackets
            '[': 16, ']': 17,  # Square brackets
            '=': 18,  # Double bond
            '#': 19,  # Triple bond
            ':': 20,  # Aromatic bond
            '+': 21, '-': 22,  # Charges
            '.': 23,  # Disconnected structures
            '/': 24, '\\': 25,  # Stereochemistry
            '@': 26,  # Chirality
            '*': 27,  # Wildcard/any atom
            '1': 28, '2': 29, '3': 30, '4': 31, '5': 32,  # Ring numbers
            '6': 33, '7': 34, '8': 35, '9': 36
        }
    
        # Create reverse mapping for decoding
        self.idx_to_token = {v: k for k, v in self.atom_vocab.items()}
        self.target_batch_size = batch_size
        self.sample_buffer = deque()
        self.interface_cutoff = interface_cutoff
        
        super().__init__(
            dataset,
            batch_size=batch_size * 2,  # Request more samples to handle Nones
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            drop_last=False,  # Changed to False to get all samples
            **kwargs
        )

    def get_interface_residues(self, protein_coords, ligand_coords, protein_residue_indices):
        """
        Identify protein residues within cutoff distance of any ligand atom
        
        Args:
            protein_coords: (N, 3) tensor of protein atom coordinates
            ligand_coords: (M, 3) tensor of ligand atom coordinates
            protein_residue_indices: (N,) tensor mapping each protein atom to its residue index
        
        Returns:
            interface_mask: Boolean mask of interface residues
        """
        if protein_coords.size(0) == 0 or ligand_coords.size(0) == 0:
            raise ValueError(
            f"Empty coordinates detected: protein_coords shape={protein_coords.shape}, "
                f"ligand_coords shape={ligand_coords.shape}"
            )
        # Calculate pairwise distances between all protein and ligand atoms
        protein_coords = protein_coords.unsqueeze(1)  # (N, 1, 3)
        ligand_coords = ligand_coords.unsqueeze(0)    # (1, M, 3)
        distances = torch.sqrt(((protein_coords - ligand_coords) ** 2).sum(dim=2))  # (N, M)
        
        # Find minimum distance from each protein atom to any ligand atom
        min_distances = distances.min(dim=1)[0]  # (N,)
        
        # Mark atoms within cutoff as interface atoms
        interface_atoms = min_distances < self.interface_cutoff
        
        # Convert atom-level interface to residue-level
        unique_residues = torch.unique(protein_residue_indices)
        interface_residues = torch.zeros(len(unique_residues), dtype=torch.bool)
        
        for i, res_idx in enumerate(unique_residues):
            res_atoms = protein_residue_indices == res_idx
            if interface_atoms[res_atoms].any():
                interface_residues[i] = True
                
        return interface_residues        
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

            if data['ligand'].pos.size(0) == 0:
                print(f"Empty ligand coords for {data.complex_name}")
                print(data['ligand'].smiles)
                continue

            interface_mask = self.get_interface_residues(
                data['protein'].pos,
                data['ligand'].pos,
                data['protein'].residue_indices  # You'll need to ensure this exists in your data
            )
            data['protein'].interface_mask = interface_mask
            
            # Tokenize SMILES using the BPE tokenizer
            smiles_tokens = self.smiles_tokenizer.encode(data['ligand'].smiles)
            data['ligand'].smiles_tokens = torch.tensor(smiles_tokens, dtype=torch.long)

        self.sample_buffer.extend(valid_data)

    def __iter__(self):
        self.sample_buffer.clear()  # Clear buffer at start of iteration
        
        # Create an infinite iterator over the dataset
        infinite_iterator = itertools.cycle(super().__iter__())
        
        while True:
            batch = next(infinite_iterator)
            if batch is not None:
                yield batch