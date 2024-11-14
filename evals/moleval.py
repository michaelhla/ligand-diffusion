import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
from rdkit import Chem
from rdkit.Chem import AllChem
import mdtraj as md
import pandas as pd
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
import logging
from scipy.spatial.distance import cdist

class MolecularEvaluator:
    def __init__(self, protein_targets, binding_sites, gpu_index=0):
        """
        Initialize evaluator with protein targets and binding sites
        
        Args:
            protein_targets (dict): Dictionary of {target_name: pdb_file_path}
            binding_sites (dict): Dictionary of {target_name: list_of_residue_ids}
                Example: {"target1": [45, 46, 89, 90, 91]}
            gpu_index (int): GPU device index to use
        """
        self.protein_targets = protein_targets
        self.binding_sites = binding_sites
        self.gpu_index = gpu_index
        self.platform = mm.Platform.getPlatformByName('CUDA')
        self.platform.setPropertyDefaultValue('DeviceIndex', str(gpu_index))
        
        # Load and prepare protein targets
        self.prepared_targets = {}
        for name, pdb_path in protein_targets.items():
            if name not in binding_sites:
                raise ValueError(f"Binding site not specified for target {name}")
            self.prepared_targets[name] = self._prepare_protein(pdb_path, binding_sites[name])
            
        logging.info(f"Initialized evaluator with {len(protein_targets)} targets")

    def _prepare_protein(self, pdb_path, binding_site_residues):
        """Prepare protein structure and extract binding site information"""
        pdb = app.PDBFile(pdb_path)
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
        
        # Create system with implicit solvent
        modeller = app.Modeller(pdb.topology, pdb.positions)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            implicitSolvent=app.OBC2
        )
        
        # Extract binding site information
        binding_site_atoms = []
        binding_site_coords = []
        
        for residue in modeller.topology.residues():
            if residue.index in binding_site_residues:
                # Get CA atom for center calculation
                ca_atom = next(atom for atom in residue.atoms() if atom.name == 'CA')
                binding_site_atoms.append(ca_atom)
                
                # Store coordinates of all heavy atoms in the residue
                for atom in residue.atoms():
                    if atom.element.atomic_number > 1:  # Skip hydrogens
                        binding_site_coords.append(
                            modeller.positions[atom.index].value_in_unit(unit.angstrom)
                        )
        
        binding_site_coords = np.array(binding_site_coords)
        binding_site_center = np.mean(binding_site_coords, axis=0)
        
        return {
            'pdb': pdb,
            'system': system,
            'modeller': modeller,
            'positions': modeller.positions,
            'binding_site_residues': binding_site_residues,
            'binding_site_atoms': binding_site_atoms,
            'binding_site_coords': binding_site_coords,
            'binding_site_center': binding_site_center
        }

    def _analyze_binding_pose(self, ligand_positions, target_info, ligand_topology):
        """
        Analyze the binding pose relative to the desired binding site
        
        Returns dict with:
        - distance_to_site: Distance from ligand center to binding site center
        - residue_contacts: Dict of closest distances to each binding site residue
        - in_binding_site: Boolean indicating if ligand is in binding site
        """
        # Convert ligand positions to numpy array
        ligand_coords = np.array([pos.value_in_unit(unit.angstrom) for pos in ligand_positions])
        ligand_center = np.mean(ligand_coords, axis=0)
        
        # Calculate distance to binding site center
        distance_to_site = np.linalg.norm(
            ligand_center - target_info['binding_site_center']
        )
        
        # Calculate minimum distances to each binding site residue
        residue_contacts = {}
        for residue_idx in target_info['binding_site_residues']:
            residue_coords = target_info['binding_site_coords']
            min_distance = np.min(cdist(ligand_coords, residue_coords))
            residue_contacts[residue_idx] = min_distance
        
        # Define if ligand is in binding site (within 5Å of any binding site residue)
        in_binding_site = any(dist < 5.0 for dist in residue_contacts.values())
        
        # Calculate contact scores
        contact_score = sum(1 for dist in residue_contacts.values() if dist < 4.0)
        
        return {
            'distance_to_site': distance_to_site,
            'residue_contacts': residue_contacts,
            'in_binding_site': in_binding_site,
            'contact_score': contact_score
        }

    def evaluate_molecule(self, mol, target_name):
        """
        Evaluate a single molecule against a specific target
        
        Args:
            mol: RDKit molecule object
            target_name: Name of target in self.prepared_targets
            
        Returns:
            dict: Evaluation metrics including binding energy and pose analysis
        """
        target = self.prepared_targets[target_name]
        
        # Generate 3D conformation for ligand
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Create combined system
        modeller = app.Modeller(target['pdb'].topology, target['pdb'].positions)
        
        # Add ligand to system
        ligand_system = self._prepare_ligand_system(mol)
        modeller.add(ligand_system['topology'], ligand_system['positions'])
        
        # Setup and run simulation
        simulation = self._setup_simulation(
            target['system'],
            modeller.positions
        )
        
        # Calculate binding energy
        state = simulation.context.getState(
            getEnergy=True,
            getPositions=True
        )
        
        binding_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        final_positions = state.getPositions(asNumpy=True)
        
        # Extract ligand positions from final state
        n_protein_atoms = len(target['pdb'].positions)
        ligand_positions = final_positions[n_protein_atoms:]
        
        # Analyze binding pose
        pose_analysis = self._analyze_binding_pose(
            ligand_positions,
            target,
            ligand_system['topology']
        )
        
        return {
            'binding_energy': binding_energy,
            'distance_to_site': pose_analysis['distance_to_site'],
            'in_binding_site': pose_analysis['in_binding_site'],
            'contact_score': pose_analysis['contact_score'],
            'residue_contacts': pose_analysis['residue_contacts'],
            'target': target_name
        }

    def _setup_simulation(self, system, positions, steps=5000):
        """Setup OpenMM simulation"""
        integrator = mm.LangevinMiddleIntegrator(
            300*unit.kelvin,
            1/unit.picosecond,
            0.002*unit.picoseconds
        )
        
        simulation = app.Simulation(
            system.topology,
            system,
            integrator,
            self.platform
        )
        simulation.context.setPositions(positions)
        
        # Minimize and equilibrate
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        simulation.step(steps)
        
        return simulation
    
    def _prepare_ligand_system(self, mol):
        """Prepare ligand system for simulation"""
        with tempfile.NamedTemporaryFile(suffix='.pdb') as tmp:
            writer = Chem.PDBWriter(tmp.name)
            writer.write(mol)
            writer.close()
            
            # Load ligand with OpenMM
            pdb = app.PDBFile(tmp.name)
            
        return {
            'topology': pdb.topology,
            'positions': pdb.positions
        }
    
    def batch_evaluate(self, molecules, n_workers=4):
        """
        Evaluate multiple molecules against all targets in parallel
        
        Args:
            molecules: List of RDKit molecules
            n_workers: Number of parallel workers
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for mol in molecules:
                for target_name in self.protein_targets:
                    futures.append(
                        executor.submit(
                            self.evaluate_molecule,
                            mol,
                            target_name
                        )
                    )
            
            for future in futures:
                results.append(future.result())
                
        return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Example protein targets and binding sites
    targets = {
        "target1": "path/to/target1.pdb",
        "target2": "path/to/target2.pdb"
    }
    
    binding_sites = {
        "target1": [45, 46, 89, 90, 91],  # Residue IDs defining binding site
        "target2": [123, 124, 125, 150, 151]
    }
    
    # Initialize evaluator
    evaluator = MolecularEvaluator(targets, binding_sites, gpu_index=0)
    
    # Example molecules (replace with your generated molecules)
    molecules = [
        Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"),
        Chem.MolFromSmiles("CC1=CC=C(C=C1)NC(=O)CN2C=CN=C2")
    ]
    
    # Run batch evaluation
    results = evaluator.batch_evaluate(molecules)
    print(results)