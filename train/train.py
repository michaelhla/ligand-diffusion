from datasets.pdbbind import PDBBind
from datasets import MOAD
from datasets.dataloader import ProteinLigandDataLoader
from torch.utils.data import ConcatDataset
import torch
from pathlib import Path
from model.smiles_tokenizer import SMILESBPETokenizer


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

def prepare_tokenizer(smiles_list, vocab_size=1000, save_dir="tokenizers"):
    """
    Prepare or load a SMILES tokenizer
    
    Args:
        smiles_list: List of all SMILES strings in your dataset
        vocab_size: Maximum vocabulary size for BPE
        save_dir: Directory to save/load tokenizer
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    tokenizer_path = save_dir / "smiles_tokenizer.json"
    
    # If tokenizer already exists, load it
    if tokenizer_path.exists():
        print("Loading existing tokenizer...")
        tokenizer = SMILESBPETokenizer.load(tokenizer_path)
    else:
        print("Training new tokenizer...")
        tokenizer = SMILESBPETokenizer(vocab_size=vocab_size)
        tokenizer.train(smiles_list)
        tokenizer.save(tokenizer_path)
        
    return tokenizer


def train_step(model, batch, optimizer):
    # Get random timestep
    t = torch.randint(0, model.diffusion.num_steps, (batch.size(0),))
    
    # Forward pass
    pred_logits = model(
        protein_data=batch['protein'],
        ligand_tokens=batch['ligand'].smiles_tokens,
        t=t
    )
    
    # Calculate loss
    loss = F.cross_entropy(
        pred_logits.view(-1, pred_logits.size(-1)),
        batch['ligand'].smiles_tokens.view(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

all_smiles = []
for data in combined_dataset:
all_smiles.append(data['smiles'])

# Prepare tokenizer
tokenizer = prepare_tokenizer(all_smiles)

# Create dataloader with tokenizer
train_loader = ProteinLigandDataLoader(
    dataset=dataset,
    batch_size=32,
    smiles_tokenizer=tokenizer
)

# Use in training
for batch in train_loader:
    # batch.protein.pos - protein coordinates
    # batch.protein.residues - protein residues
    # batch.ligand.pos - ligand coordinates
    # batch.ligand.atom_types - ligand atom types
    # batch.ligand.smiles - ligand SMILES strings
    pass