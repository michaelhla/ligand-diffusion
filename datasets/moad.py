from typing import Optional, List
import os
from torch_geometric.data import HeteroData
from protein_ligand import ProteinLigandDataset
import numpy as np
from rdkit import Chem
import torch
import rdkit.RDLogger as RDLogger


# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

class MOAD(ProteinLigandDataset):
    def get_complex_list(self) -> List[str]:
        print("Getting MOAD complex list...")
        protein_dir = os.path.join(self.root, 'pdb_protein')
        ligand_dir = os.path.join(self.root, 'pdb_superligand')
        
        # Get first few ligand files for testing
        complex_pairs = []
        for ligand_file in sorted(os.listdir(ligand_dir)):
            # Skip hidden files and system files
            if ligand_file.startswith('.'):
                continue
                
            if ligand_file.endswith('_superlig_0.pdb'):
                protein_id = ligand_file.split('_superlig_')[0]
                protein_file = os.path.join(protein_dir, f"{protein_id}_protein.pdb")
                
                # Verify both protein and ligand files exist
                if not os.path.exists(protein_file):
                    print(f"Skipping {protein_id}: protein file not found")
                    continue
                    
                # Count valid ligand files
                ligand_count = 0
                for i in range(10):  # Reasonable upper limit for ligands
                    test_ligand = os.path.join(ligand_dir, f"{protein_id}_superlig_{i}.pdb")
                    if not os.path.exists(test_ligand):
                        break
                    ligand_count += 1
                    
                for lig_idx in range(ligand_count):
                    complex_pairs.append((protein_id, lig_idx))
                    
                if len(complex_pairs) >= 320:  # Limit to ~10 protein-ligand pairs
                    break
        
        print(f"Found {len(complex_pairs)} valid MOAD complexes for testing")
        for pair in complex_pairs:
            print(f"  - {pair[0]} (ligand {pair[1]})")
        return complex_pairs

    def process_complex(self, complex_pair: tuple) -> Optional[HeteroData]:
        try:
            protein_id, ligand_idx = complex_pair
            print(f"Processing MOAD complex: {protein_id} ligand {ligand_idx}")
            
            # Create paths
            protein_file = os.path.join(self.root, 'pdb_protein', f"{protein_id}_protein.pdb")
            ligand_file = os.path.join(self.root, 'pdb_superligand', 
                                      f"{protein_id}_superlig_{ligand_idx}.pdb")
            
            if not os.path.exists(protein_file) or not os.path.exists(ligand_file):
                print(f"Files not found for {protein_id} ligand {ligand_idx}")
                return None
                
            # Process protein and ligand
            protein_coords, residue_names = self.process_protein(protein_file)
            ligand_data = self.process_ligand(ligand_file)
            if ligand_data is None:
                return None
                
            ligand_coords, atom_types, smiles = ligand_data
            
            # Create graph data - only convert coordinates to tensors
            data = HeteroData()
            data['protein'].pos = torch.from_numpy(protein_coords).float()
            data['protein'].residues = residue_names  # Keep as strings
            data['ligand'].pos = torch.from_numpy(ligand_coords).float()
            data['ligand'].atom_types = atom_types  # Keep as strings
            data['ligand'].smiles = smiles
            data.complex_name = f"{protein_id}_ligand_{ligand_idx}"
            
            print(f"Successfully created HeteroData for {protein_id} ligand {ligand_idx}")
            return data
            
        except Exception as e:
            print(f"Error processing {protein_id} with ligand {ligand_idx}: {str(e)}")
            return None
        
    def process_protein(self, protein_file: str) -> tuple:
        """Extract protein coordinates and residue information from PDB file.
        
        Args:
            protein_file: Path to protein PDB file
            
        Returns:
            tuple: (protein_coords, protein_residues)
                protein_coords: numpy array of shape (N, 3) containing atom coordinates
                protein_residues: list of residue names
        """
        coords = []
        residues = []
        
        with open(protein_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    # Extract coordinates
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                    
                    # Extract residue name
                    residue = line[17:20].strip()
                    residues.append(residue)
                    
        return np.array(coords, dtype=np.float32), residues

    def process_ligand(self, ligand_file: str) -> tuple:
        """Extract ligand information from PDB file.
        
        Args:
            ligand_file: Path to ligand PDB file
            
        Returns:
            tuple: (ligand_coords, atom_types, smiles)
                ligand_coords: numpy array of shape (N, 3) containing atom coordinates
                atom_types: list of atom types
                smiles: SMILES string representation of ligand
        """
        coords = []
        atom_types = []
        
        with open(ligand_file, 'r') as f:
            for line in f:
                if line.startswith('HETATM'):
                    # Extract coordinates
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip()) 
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                    
                    # Extract atom type
                    atom = line[76:78].strip()
                    atom_types.append(atom)
        
        # Convert to RDKit mol and get SMILES
        mol = Chem.MolFromPDBFile(ligand_file, removeHs=False, sanitize=True)
        if mol is None:
            return None
            
        # Force conformer to be recognized as 3D
        for conf in mol.GetConformers():
            conf.Set3D(True)
            
        smiles = Chem.MolToSmiles(mol)
            
        # Convert coordinates to numpy array
        coords = np.array(coords, dtype=np.float32)
        
        return coords, atom_types, smiles