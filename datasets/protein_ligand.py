import os
import pickle
from multiprocessing import Pool
from typing import Optional, List, Dict

import torch
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm
from rdkit import Chem
import numpy as np

class ProteinLigandDataset(Dataset):
    # Add residue vocabulary
    RESIDUE_VOCAB = {
        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
        'UNK': 20  # Unknown residue type
    }

    def __init__(
        self,
        root: str,
        transform=None,
        split: str = 'train',
        num_workers: int = 1,
        cache_path: Optional[str] = None,
        precompute: bool = False
    ):
        super().__init__(root, transform)
        self.split = split
        self.num_workers = num_workers
        self.cache_path = cache_path
        self.precompute = precompute
        
        # Get list of complexes
        self.complex_list = self.get_complex_list()
        
        # Initialize cache dict if using caching
        self.cache: Dict[str, HeteroData] = {}
        
        if precompute:
            if cache_path:
                os.makedirs(cache_path, exist_ok=True)
                if self.check_cached_data():
                    self.cache = self.load_cached_data()
                else:
                    self.precompute_all()
                    self.save_cached_data()
            else:
                self.precompute_all()

    def precompute_all(self):
        """Process all complexes in parallel if precomputing."""
        if self.num_workers > 1:
            with Pool(self.num_workers) as p:
                with tqdm(total=len(self.complex_list), desc=f'Processing {self.split} data') as pbar:
                    for name, data in p.imap_unordered(self.process_complex_wrapper, self.complex_list):
                        if data is not None:
                            self.cache[name] = data
                        pbar.update()
        else:
            for complex_name in tqdm(self.complex_list, desc=f'Processing {self.split} data'):
                data = self.process_complex(complex_name)
                if data is not None:
                    self.cache[complex_name] = data

    def process_complex_wrapper(self, complex_name):
        """Wrapper to return both name and processed data for parallel processing."""
        return complex_name, self.process_complex(complex_name)

    def check_cached_data(self) -> bool:
        """Check if processed data exists in cache."""
        return os.path.exists(os.path.join(self.cache_path, f"{self.split}_data.pkl"))

    def load_cached_data(self) -> Dict[str, HeteroData]:
        """Load processed data from cache."""
        with open(os.path.join(self.cache_path, f"{self.split}_data.pkl"), 'rb') as f:
            return pickle.load(f)

    def save_cached_data(self):
        """Save processed data to cache."""
        if self.cache_path:
            with open(os.path.join(self.cache_path, f"{self.split}_data.pkl"), 'wb') as f:
                pickle.dump(self.cache, f)

    def len(self):
        return len(self.complex_list)

    def get(self, idx):
        complex_name = self.complex_list[idx]
        
        # If precomputed or cached, return from cache
        if complex_name in self.cache:
            return self.cache[complex_name]
            
        # Otherwise, process on the fly
        data = self.process_complex(complex_name)
        
        # Optionally cache the result
        if self.cache_path and data is not None:
            self.cache[complex_name] = data
            
        return data

    def get_complex_list(self) -> List[str]:
        """Get list of complex names to process. Implement in child class."""
        raise NotImplementedError

    def process_complex(self, complex_name: str) -> Optional[HeteroData]:
        """Process a single complex. Implement in child class."""
        raise NotImplementedError

    def process_protein(self, protein_file: str) -> tuple:
        """Extract backbone atoms (N, CA, C) from PDB file.
        
        Args:
            protein_file: Path to protein PDB file
            
        Returns:
            tuple: (coords, residue_names, residue_indices)
                coords: numpy array of shape (num_residues, 3, 3) containing backbone atom coordinates
                residue_names: list of residue names
                residue_indices: numpy array of residue indices
        """
        backbone_atoms = ['N', 'CA', 'C']
        current_residue = None
        residue_coords = []
        residue_names = []
        residue_indices = []
        current_bb_atoms = {}
        
        with open(protein_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    residue_num = int(line[22:26].strip())
                    
                    # Only process backbone atoms
                    if atom_name not in backbone_atoms:
                        continue
                        
                    # Extract coordinates
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    # If we're starting a new residue
                    if residue_num != current_residue:
                        # Store the previous residue if we have all backbone atoms
                        if current_residue is not None and len(current_bb_atoms) == 3:
                            coords = [current_bb_atoms[atom] for atom in backbone_atoms]
                            residue_coords.append(coords)
                            residue_names.append(current_residue_name)
                            residue_indices.append(current_residue)
                        
                        # Start new residue
                        current_residue = residue_num
                        current_residue_name = residue_name
                        current_bb_atoms = {}
                    
                    # Store this backbone atom
                    current_bb_atoms[atom_name] = [x, y, z]
        
        # Don't forget to store the last residue
        if current_residue is not None and len(current_bb_atoms) == 3:
            coords = [current_bb_atoms[atom] for atom in backbone_atoms]
            residue_coords.append(coords)
            residue_names.append(current_residue_name)
            residue_indices.append(current_residue)
        
        # Convert to numpy arrays
        coords = np.array(residue_coords, dtype=np.float32)  # Shape: (num_residues, 3, 3)
        residue_indices = np.array(residue_indices, dtype=np.int64)
        
        return coords, residue_names, residue_indices

    def process_ligand(self, ligand_file: str) -> tuple:
        """Extract ligand information and create SMILES string. Implement in child class."""
        raise NotImplementedError