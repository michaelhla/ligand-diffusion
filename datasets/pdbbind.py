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

class PDBBind(ProteinLigandDataset):
    def get_complex_list(self) -> List[str]:
        print("Getting PDBBind complex list...")
        complex_dirs = os.listdir(self.root)
        # Take only first few directories for quick testing
        complex_dirs = complex_dirs[:160]  # Limit to 10 complexes
        print(f"Using {len(complex_dirs)} PDBBind complexes for testing")
        return [d for d in complex_dirs if os.path.isdir(os.path.join(self.root, d))]

    def process_complex(self, complex_name: str) -> Optional[HeteroData]:
        try:
            print(f"Processing PDBBind complex: {complex_name}")
            protein_file = os.path.join(self.root, complex_name, f"{complex_name}_protein_processed.pdb")
            ligand_file = os.path.join(self.root, complex_name, f"{complex_name}_ligand.sdf")
            
            if not os.path.exists(protein_file) or not os.path.exists(ligand_file):
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
            data.complex_name = complex_name
            
            return data
            
        except Exception as e:
            print(f"Error processing {complex_name}: {str(e)}")
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
                    
        return np.array(coords), residues

    def process_ligand(self, ligand_file: str) -> tuple:
        """Extract ligand information from SDF file.
        
        Args:
            ligand_file: Path to ligand SDF file
            
        Returns:
            tuple: (ligand_coords, atom_types, smiles)
                ligand_coords: numpy array of shape (N, 3) containing atom coordinates
                atom_types: list of atom types
                smiles: SMILES string representation of ligand
        """
        # Read SDF file with RDKit, forcing 3D conformer reading
        mol_supplier = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=True)
        mol = mol_supplier[0]
        if mol is None:
            return None
            
        # Force conformer to be recognized as 3D
        for conf in mol.GetConformers():
            conf.Set3D(True)
            
        # Get coordinates and atom types
        conf = mol.GetConformer()
        coords = []
        atom_types = []
        
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            atom = mol.GetAtomWithIdx(i)
            atom_types.append(atom.GetSymbol())
            
        # Generate SMILES
        smiles = Chem.MolToSmiles(mol)
            
        # Convert to numpy arrays before returning
        coords = np.array(coords, dtype=np.float32)
        
        return coords, atom_types, smiles