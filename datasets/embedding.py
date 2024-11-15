import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class EmbeddingDataset(Dataset):
    def __init__(self, h5_paths, smiles_tokenizer):
        """
        Args:
            h5_paths: List of paths to H5 files containing embeddings
            smiles_tokenizer: Tokenizer for SMILES strings
        """
        self.tokenizer = smiles_tokenizer
        self.complex_names = []
        self.h5_files = []
        
        # Open all H5 files and keep them open
        for path in h5_paths:
            h5_file = h5py.File(path, 'r')
            self.h5_files.append(h5_file)
            
            # Get complex names from this file
            self.complex_names.extend([
                (len(self.h5_files)-1, name) 
                for name in h5_file['full_protein'].keys()
            ])
    
    def __len__(self):
        return len(self.complex_names)
    
    def __getitem__(self, idx):
        file_idx, complex_name = self.complex_names[idx]
        h5_file = self.h5_files[file_idx]
        
        # Get protein embeddings and SMILES
        protein_embed = torch.from_numpy(h5_file['full_protein'][complex_name][()])
        smiles = h5_file['smiles'][complex_name][()]  # Assuming SMILES are stored in H5
        
        # Tokenize SMILES
        smiles_tokens = torch.tensor(self.tokenizer.encode(smiles), dtype=torch.long)
        
        return {
            'protein_embeddings': protein_embed,
            'smiles_tokens': smiles_tokens,
            'complex_name': complex_name
        }
    
    def __del__(self):
        # Close H5 files when dataset is destroyed
        for h5_file in self.h5_files:
            h5_file.close()

def collate_embeddings(batch):
    """Custom collate function for batching"""
    if not batch:
        return None
        
    # Get max lengths
    max_protein_len = max(x['protein_embeddings'].size(0) for x in batch)
    max_smiles_len = max(x['smiles_tokens'].size(0) for x in batch)
    
    # Prepare tensors
    protein_embeddings = torch.zeros(len(batch), max_protein_len, 1536)
    smiles_tokens = torch.zeros(len(batch), max_smiles_len, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        p_len = item['protein_embeddings'].size(0)
        s_len = item['smiles_tokens'].size(0)
        
        protein_embeddings[i, :p_len] = item['protein_embeddings']
        smiles_tokens[i, :s_len] = item['smiles_tokens']
    
    return {
        'protein_embeddings': protein_embeddings,
        'smiles_tokens': smiles_tokens
    }