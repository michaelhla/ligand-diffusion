import os
import glob
from pathlib import Path
import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pickle
from tqdm import tqdm

## goal is to persist the dataset in a format that is easy to load and use for training our model 
## we want 3d coords of the target protein, 3d coords of a selected binding interface, and smiles string
## (coords_prot, coords_interface, coords_ligand, smiles)

@dataclass
class MOADExample:
    """Single example from BindingMOAD dataset"""
    pdb_id: str
    protein_file: str      # Path to protein PDB file
    ligand_file: str       # Path to ligand PDB file
    ligand_smiles: str     # SMILES string of ligand
    ligand_number: int     # Index of ligand for this protein

def parse_protein(structure) -> Tuple[np.ndarray, List[str]]:
    """Extract coordinates and atom types from protein structure"""
    coords = []
    atoms = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
                    atoms.append(atom.element)
                    
    return np.array(coords), atoms

def parse_ligand(mol) -> Tuple[np.ndarray, str]:
    """Extract 3D coordinates and SMILES from ligand"""
    # Get 3D coordinates
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    
    # Generate SMILES
    smiles = Chem.MolToSmiles(mol)
    
    return coords, smiles

def process_single_protein(protein_file: str, ligand_dir: str) -> List[MOADExample]:
    """Process a single protein file and its ligands"""
    examples = []
    pdb_id = Path(protein_file).stem[:4]  # First 4 chars are PDB ID
    
    try:
        # Find all corresponding ligand files
        ligand_pattern = os.path.join(ligand_dir, f"{pdb_id}_*_superlig_*.pdb")
        ligand_files = glob.glob(ligand_pattern)
        
        if not ligand_files:
            print(f"No ligand files found for {pdb_id} using pattern: {ligand_pattern}")
            return []
            
        print(f"Found {len(ligand_files)} ligand files for {pdb_id}")
        
        for ligand_file in ligand_files:
            try:
                # Extract ligand number from filename
                ligand_number = int(Path(ligand_file).stem.split('_')[-1])
                
                # Only parse ligand for SMILES
                mol = Chem.MolFromPDBFile(ligand_file, removeHs=False, sanitize=False)
                if mol is None:
                    print(f"Failed to parse ligand: {ligand_file}")
                    continue
                    
                smiles = Chem.MolToSmiles(mol)
                
                example = MOADExample(
                    pdb_id=pdb_id,
                    protein_file=protein_file,
                    ligand_file=ligand_file,
                    ligand_smiles=smiles,
                    ligand_number=ligand_number
                )
                examples.append(example)
                
            except Exception as e:
                print(f"Error processing ligand {ligand_file}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error processing protein {pdb_id}: {str(e)}")
    
    return examples

def process_batch(protein_files: List[str], ligand_dir: str, batch_size: int = 100) -> List[MOADExample]:
    """Process a batch of protein files"""
    batch_examples = []
    
    for protein_file in protein_files:
        examples = process_single_protein(protein_file, ligand_dir)
        if examples:
            batch_examples.extend(examples)
            
    return batch_examples

def load_moad_dataset(protein_dir: str, ligand_dir: str, batch_size: int = 100, cache_file: str = None) -> List[MOADExample]:
    """Load and parse BindingMOAD dataset using batching"""
    # # Try to load from cache first
    # if cache_file and os.path.exists(cache_file):
    #     print(f"Loading from cache: {cache_file}")
    #     return load_dataset(cache_file)
    
    # Find all protein PDB files
    protein_files = glob.glob(os.path.join(protein_dir, "*.pdb"))
    print(f"\nFound {len(protein_files)} protein files in {protein_dir}")
    
    # Process files in batches
    all_examples = []
    num_batches = (len(protein_files) + batch_size - 1) // batch_size  # Round up division
    
    for i in tqdm(range(num_batches), desc=f"Processing proteins in batches of {batch_size}"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(protein_files))
        batch_files = protein_files[start_idx:end_idx]
        
        print(f"\nProcessing batch {i+1}/{num_batches} ({len(batch_files)} files)")
        batch_examples = process_batch(batch_files, ligand_dir)
        print(f"Found {len(batch_examples)} examples in this batch")
        
        if batch_examples:
            all_examples.extend(batch_examples)
            
        # Optionally save intermediate results
        if cache_file and (i + 1) % 10 == 0:  # Save every 10 batches
            save_dataset(all_examples, cache_file + f".batch_{i}")
    
    # Save final results to cache if specified
    if cache_file:
        save_dataset(all_examples, cache_file)
        
    return all_examples

def save_dataset(dataset: List[MOADExample], output_path: str):
    """Save dataset to disk using pickle"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(input_path: str) -> List[MOADExample]:
    """Load dataset from disk"""
    with open(input_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    protein_dir = "/workspace/dockgen/BindingMOAD_2020_processed/pdb_protein"
    ligand_dir = "/workspace/dockgen/BindingMOAD_2020_processed/pdb_superligand"
    output_path = "/workspace/dockgen/processed_data/moad_dataset.pkl"
    
    # Process and save dataset
    dataset = load_moad_dataset(
        protein_dir=protein_dir,
        ligand_dir=ligand_dir,
        batch_size=100,  # Adjust based on your system
        cache_file=output_path
    )
    print(f"Processed {len(dataset)} examples from BindingMOAD")
    
    # Print dimensions of first example
    if len(dataset) > 0:
        example = dataset[0]
        print("\nDimensions of first example:")
        print(f"Protein coordinates shape: {example.protein_coords.shape}")
        print(f"Number of protein atoms: {len(example.protein_atoms)}")
        print(f"Ligand coordinates shape: {example.ligand_coords.shape}")
        print(f"SMILES string length: {len(example.ligand_smiles)}")
        print(f"Example SMILES: {example.ligand_smiles}")
    
    save_dataset(dataset, output_path)
    print(f"\nSaved dataset to {output_path}")
    
    # Test loading
    loaded_dataset = load_dataset(output_path)
    print(f"Successfully loaded {len(loaded_dataset)} examples")
