from typing import Optional, List
import os
from torch_geometric.data import HeteroData
from .protein_ligand import ProteinLigandDataset
import numpy as np
from rdkit import Chem

class MOAD(ProteinLigandDataset):
    def get_complex_list(self) -> List[str]:
        # Implement MOAD-specific complex list generation
        # This might involve reading from cluster files or other MOAD-specific organization
        pass

    def process_complex(self, complex_name: str) -> Optional[HeteroData]:
        try:
            # Create paths (adjust paths according to MOAD structure)
            protein_file = os.path.join(self.root, 'pdb_protein', f"{complex_name}_protein.pdb")
            ligand_file = os.path.join(self.root, 'pdb_ligand', f"{complex_name}_ligand.sdf")
            
            # Process protein
            protein_coords, protein_residues = self.process_protein(protein_file)
            
            # Process ligand
            ligand_data = self.process_ligand(ligand_file)
            if ligand_data is None:
                return None
            ligand_coords, atom_types, smiles = ligand_data
            
            # Create graph data
            data = HeteroData()
            data['protein'].pos = protein_coords
            data['protein'].residues = protein_residues
            data['ligand'].pos = ligand_coords
            data['ligand'].atom_types = atom_types
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
        mol = Chem.MolFromPDBFile(ligand_file)
        if mol is None:
            return None
        smiles = Chem.MolToSmiles(mol)
            
        return np.array(coords), atom_types, smiles