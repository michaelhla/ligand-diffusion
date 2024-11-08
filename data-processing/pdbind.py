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
class PDBindExample:
    """Single example from PDBind dataset"""
    pdb_id: str
    protein_coords: np.ndarray  # (N, 3) array of protein atom coordinates
    protein_atoms: List[str]    # List of protein atom types
    ligand_coords: np.ndarray   # (M, 3) array of ligand atom coordinates  
    ligand_smiles: str         # SMILES string of ligand
    binding_pocket: Optional[np.ndarray] = None  # To be implemented

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

def load_pdbind_dataset(data_dir: str) -> List[PDBindExample]:
    """Load and parse PDBind dataset"""
    examples = []
    parser = PDB.PDBParser(QUIET=True)
    
    # Find all PDB files in directory
    pdb_files = glob.glob(os.path.join(data_dir, "**/*.pdb"), recursive=True)
    
    for pdb_file in pdb_files:
        pdb_id = Path(pdb_file).stem
        
        try:
            # Load protein structure
            structure = parser.get_structure(pdb_id, pdb_file)
            protein_coords, protein_atoms = parse_protein(structure)
            
            # Load ligand structure 
            # Assuming ligand file is in same directory with _ligand.pdb suffix
            ligand_file = pdb_file.replace(".pdb", "_ligand.pdb")
            if not os.path.exists(ligand_file):
                continue
                
            mol = Chem.MolFromPDBFile(ligand_file, removeHs=False)
            if mol is None:
                continue
                
            ligand_coords, ligand_smiles = parse_ligand(mol)
            
            example = PDBindExample(
                pdb_id=pdb_id,
                protein_coords=protein_coords,
                protein_atoms=protein_atoms,
                ligand_coords=ligand_coords,
                ligand_smiles=ligand_smiles
            )
            examples.append(example)
            
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            continue
            
    return examples

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/pdbind/data"
    dataset = load_pdbind_dataset(data_dir)
    print(f"Loaded {len(dataset)} examples from PDBind")
