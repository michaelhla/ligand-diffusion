import torch
import os
from tqdm import tqdm
import h5py
from torch.utils.data import DataLoader
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.esm import ProteinEncoder
from datasets.pdbbind import PDBBind
from datasets.moad import MOAD
from datasets.dataloader import ProteinLigandDataLoader
import hashlib
from huggingface_hub import login 

login(token=open("api_keys.txt").readlines()[0].split("hf: ")[1].strip())


def get_protein_hash(protein_coords):
    """
    Generate a unique hash for a protein structure based on its coordinates
    """
    return hashlib.sha256(protein_coords.numpy().tobytes()).hexdigest()

def process_dataset(dataset, protein_encoder, batch_size=1, save_path=None):
    """
    Process a dataset and save embeddings to H5 file
    """
    device = next(protein_encoder.parameters()).device
    
    # Create basic dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x[0]
    )
    
    # Dictionary to store protein embeddings keyed by their hash
    protein_cache = {}
    
    # If file exists, load existing embeddings into cache
    if os.path.exists(save_path):
        print(f"Loading existing embeddings from {save_path}")
        with h5py.File(save_path, 'r') as f:
            if 'protein_hashes' in f:
                for complex_name in f['protein_hashes']:
                    protein_hash = f['protein_hashes'][complex_name][()].decode('utf-8')
                    if 'full_protein' in f and protein_hash not in protein_cache:
                        protein_cache[protein_hash] = torch.from_numpy(f['full_protein'][complex_name][()])
        print(f"Loaded {len(protein_cache)} unique protein embeddings from cache")
    
    with h5py.File(save_path, 'a') as f:  # Use 'a' mode to append to existing file
        # Create groups if they don't exist
        if 'full_protein' not in f:
            full_protein_group = f.create_group('full_protein')
        else:
            full_protein_group = f['full_protein']
            
        if 'interface' not in f:
            interface_group = f.create_group('interface')
        else:
            interface_group = f['interface']
            
        if 'protein_hashes' not in f:
            hash_group = f.create_group('protein_hashes')
        else:
            hash_group = f['protein_hashes']
        
        for data in tqdm(dataloader, desc=f"Processing {save_path}"):
            if data is None:
                continue
                
            try:
                complex_name = data.complex_name
                
                # Skip if this complex has already been processed
                if complex_name in full_protein_group:
                    continue
                
                # Prepare protein data
                protein_coords = data['protein'].pos
                protein_hash = get_protein_hash(protein_coords)
                
                # Get or compute protein embeddings
                if protein_hash in protein_cache:
                    embeddings = protein_cache[protein_hash]
                else:
                    residue_indices = data['protein'].residue_indices if hasattr(data['protein'], 'residue_indices') else torch.arange(len(protein_coords))
                    
                    # Prepare input for ESM (add batch dimension)
                    protein_data = {
                        'coords': protein_coords.unsqueeze(0).to(device),
                        'residue_indices': residue_indices.unsqueeze(0).to(device)
                    }
                    
                    # Get embeddings
                    with torch.no_grad():
                        output = protein_encoder(protein_data)
                        embeddings = output['embeddings'].cpu().squeeze(0)  # (num_residues, hidden_dim)
                    
                    # Cache the embeddings
                    protein_cache[protein_hash] = embeddings
                
                # Calculate interface mask
                ligand_coords = data['ligand'].pos
                residue_indices = torch.arange(len(protein_coords))
                interface_mask = ProteinLigandDataLoader.get_interface_residues(
                    protein_coords=protein_coords,
                    ligand_coords=ligand_coords,
                    residue_indices=residue_indices,
                    interface_cutoff=8.0
                )
                
                # Store full protein embeddings
                full_protein_group.create_dataset(
                    complex_name,
                    data=embeddings.numpy(),
                    compression='gzip'
                )
                
                # Store interface embeddings
                interface_embeddings = embeddings[interface_mask]
                interface_group.create_dataset(
                    complex_name,
                    data=interface_embeddings.numpy(),
                    compression='gzip'
                )
                
                # Store interface mask
                interface_group.create_dataset(
                    f"{complex_name}_mask",
                    data=interface_mask.numpy(),
                    compression='gzip'
                )
                
                # Store protein hash
                hash_group.create_dataset(
                    complex_name,
                    data=protein_hash.encode('utf-8')
                )
                
            except Exception as e:
                print(f"\nError processing sample {complex_name}: {e}")
                continue
        
        print(f"\nProcessed dataset with {len(protein_cache)} unique proteins")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize ESM encoder
    protein_encoder = ProteinEncoder().to(device)
    protein_encoder.eval()
    
    # Create output directory
    os.makedirs("data/embeddings", exist_ok=True)
    
    # Process PDBBind
    pdbbind_dataset = PDBBind(
        root='/workspace/pdbbind/PDBBind_processed',
        cache_path=None,
        split='train',
        num_workers=4,
        precompute=False
    )
    process_dataset(
        pdbbind_dataset,
        protein_encoder,
        save_path="/workspace/embeddings/pdbbind_train_embeddings.h5"
    )
    
    # Process MOAD
    moad_dataset = MOAD(
        root='/workspace/dockgen/BindingMOAD_2020_processed',
        cache_path=None,
        split='train',
        num_workers=4,
        precompute=False
    )
    process_dataset(
        moad_dataset,
        protein_encoder,
        save_path="/workspace/embeddings/moad_train_embeddings.h5"
    )

if __name__ == "__main__":
    main()