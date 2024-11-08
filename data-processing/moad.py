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

@dataclass
class MOADExample:
    """Single example from BindingMOAD dataset"""
    pdb_id: str
    protein_coords: np.ndarray  # (N, 3) array of protein atom coordinates
    protein_atoms: List[str]    # List of protein atom types
    ligand_coords: np.ndarray   # (M, 3) array of ligand atom coordinates  
    ligand_smiles: str         # SMILES string of ligand
    ligand_number: int        # Index of ligand for this protein

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

def load_moad_dataset(protein_dir: str, ligand_dir: str) -> List[MOADExample]:
    """Load and parse BindingMOAD dataset"""
    examples = []
    parser = PDB.PDBParser(QUIET=True)
    
    # Find all protein PDB files
    protein_files = glob.glob(os.path.join(protein_dir, "*.pdb"))
    
    for protein_file in protein_files:
        pdb_id = Path(protein_file).stem[:4]  # First 4 chars are PDB ID
        
        try:
            # Load protein structure
            structure = parser.get_structure(pdb_id, protein_file)
            protein_coords, protein_atoms = parse_protein(structure)
            
            # Find all corresponding ligand files
            ligand_pattern = os.path.join(ligand_dir, f"{pdb_id}_*_superlig_*.pdb")
            ligand_files = glob.glob(ligand_pattern)
            
            for ligand_file in ligand_files:
                try:
                    # Extract ligand number from filename
                    ligand_number = int(Path(ligand_file).stem.split('_')[-1])
                    
                    mol = Chem.MolFromPDBFile(ligand_file, removeHs=False)
                    if mol is None:
                        print(f"Failed to parse ligand: {ligand_file}")
                        continue
                        
                    ligand_coords, ligand_smiles = parse_ligand(mol)
                    
                    example = MOADExample(
                        pdb_id=pdb_id,
                        protein_coords=protein_coords,
                        protein_atoms=protein_atoms,
                        ligand_coords=ligand_coords,
                        ligand_smiles=ligand_smiles,
                        ligand_number=ligand_number
                    )
                    examples.append(example)
                    
                except Exception as e:
                    print(f"Error processing ligand {ligand_file}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error processing protein {pdb_id}: {e}")
            continue
            
    return examples

if __name__ == "__main__":
    protein_dir = "/workspace/dockgen/BindingMOAD_2020_processed/pdb_protein"
    ligand_dir = "/workspace/dockgen/BindingMOAD_2020_processed/pdb_superligand"
    dataset = load_moad_dataset(protein_dir, ligand_dir)
    print(f"Loaded {len(dataset)} examples from BindingMOAD")
