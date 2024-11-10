from datasets.pdbbind import PDBBind
from datasets import MOAD
from datasets.dataloader import ProteinLigandDataLoader
from torch.utils.data import ConcatDataset

# Create individual datasets
pdbbind_dataset = PDBBind(
    root='/workspace/pdbbind/PDBBind_processed',
    cache_path='/workspace/pdbbind/cache',
    split='train',
    num_workers=4
)

moad_dataset = MOAD(
    root='/workspace/dockgen/BindingMOAD_2020_processed',
    cache_path='/workspace/dockgen/cache',
    split='train',
    num_workers=4
)

# Combine datasets
combined_dataset = ConcatDataset([pdbbind_dataset, moad_dataset])

# Create dataloader with combined dataset
train_loader = ProteinLigandDataLoader(
    combined_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Use in training
for batch in train_loader:
    # batch.protein.pos - protein coordinates
    # batch.protein.residues - protein residues
    # batch.ligand.pos - ligand coordinates
    # batch.ligand.atom_types - ligand atom types
    # batch.ligand.smiles - ligand SMILES strings
    pass