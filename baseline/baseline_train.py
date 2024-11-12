import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import functional as F
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from huggingface_hub import login
import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataloader import ProteinLigandDataLoader
from baseline.baseline_model import BaselineModel
from datasets.pdbbind import PDBBind
from datasets.moad import MOAD

login(token=open("api_keys.txt").read().split(":")[1].strip())


def train():
    # Initialize wandb
    wandb.init(
        project="ligand-diffusion",
        name="baseline-test",
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "max_epochs": 100,
            "hidden_dim": 1024,
            "num_layers": 6,
            "num_heads": 8,
            "grad_clip": 1.0,
            "warmup_steps": 1000,
        }
    )
    config = wandb.config

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer (using same vocab as dataloader)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with your actual tokenizer
    
    # Initialize datasets
    pdbbind_dataset = PDBBind(
        root='/workspace/pdbbind/PDBBind_processed',
        cache_path=None,
        split='train',
        num_workers=4,
        precompute=False
    )
    
    moad_dataset = MOAD(
        root='/workspace/dockgen/BindingMOAD_2020_processed',
        cache_path=None,
        split='train',
        num_workers=4,
        precompute=False
    )

    # Print example from PDBBind dataset
    pdbbind_example = pdbbind_dataset[0]
    print("\nPDBBind Example:")
    print(f"Complex name: {pdbbind_example.complex_name}")
    print(f"Number of protein atoms: {pdbbind_example['protein'].pos.shape[0]}")
    print(f"Number of ligand atoms: {pdbbind_example['ligand'].pos.shape[0]}")
    print(f"SMILES: {pdbbind_example['ligand'].smiles}")
    print(f"First 3 residues: {pdbbind_example['protein'].residues[:3]}")

    # Print example from MOAD dataset  
    moad_example = moad_dataset[0]
    print("\nMOAD Example:")
    print(f"Complex name: {moad_example.complex_name}")
    print(f"Number of protein atoms: {moad_example['protein'].pos.shape[0]}")
    print(f"Number of ligand atoms: {moad_example['ligand'].pos.shape[0]}")
    print(f"SMILES: {moad_example['ligand'].smiles}")
    print(f"First 3 residues: {moad_example['protein'].residues[:3]}")
    
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([pdbbind_dataset, moad_dataset])
    
    # Create dataloader
    dataloader = ProteinLigandDataLoader(
        combined_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        smiles_tokenizer=tokenizer
    )
    
    # Initialize model
    model = BaselineModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        vocab_size=len(dataloader.atom_vocab)
    ).to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    global_step = 0
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Learning rate warmup
            if global_step < config.warmup_steps:
                lr = config.learning_rate * (global_step / config.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Prepare input data
            protein_data = {
                'sequence_tokens': batch['protein'].residue_tokens.to(device),
                'coords': batch['protein'].pos.to(device),
                'interface_mask': batch['protein'].interface_mask.to(device)
            }
            
            smiles_tokens = batch['ligand'].smiles_tokens.to(device)
            
            # Forward pass
            logits = model(protein_data, smiles_tokens[:, :-1])  # Remove last token for input
            
            # Calculate loss (ignore padding tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                smiles_tokens[:, 1:].contiguous().view(-1),  # Shift right for teacher forcing
                ignore_index=dataloader.atom_vocab['PAD']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Log to wandb
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
                "global_step": global_step
            })
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(dataloader)
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch": epoch
        })
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/baseline_epoch_{epoch+1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            
            # Log model checkpoint to wandb
            wandb.save(checkpoint_path)

if __name__ == "__main__":
    train()