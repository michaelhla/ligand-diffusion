import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import functional as F
import os
from tqdm import tqdm
from huggingface_hub import login
import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataloader import ProteinLigandDataLoader
from baseline.baseline_model import BaselineModel
from datasets.pdbbind import PDBBind
from datasets.moad import MOAD
from model.smiles_tokenizer import SMILESBPETokenizer
import json

login(token=open("api_keys.txt").readlines()[0].split("hf: ")[1].strip())
wandb.login(key=open("api_keys.txt").readlines()[1].split("wandb: ")[1].strip())


def train():
    # Initialize wandb
    wandb.init(
        project="ligand-diffusion",
        name="baseline-test",
        config={
            "learning_rate": 1e-4,
            "batch_size": 16,
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
    
    # Initialize SMILES tokenizer
    tokenizer = SMILESBPETokenizer(vocab_size=1000)
    
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

    print(f"PDBBind dataset size: {len(pdbbind_dataset)}")
    print(f"MOAD dataset size: {len(moad_dataset)}")

    # Check for existing tokenizer checkpoint
    tokenizer_path = "checkpoints/smiles_tokenizer.json"
    if os.path.exists(tokenizer_path):
        print("Loading existing tokenizer from checkpoint...")
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
            tokenizer.vocab = tokenizer_data['vocab'] 
            tokenizer.merges = tokenizer_data['merges']
            tokenizer.special_tokens = tokenizer_data['special_tokens']
            tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    else:
        print("Training new tokenizer...")
        # Pass datasets directly to tokenizer
        tokenizer.train([pdbbind_dataset, moad_dataset])
    # Save trained tokenizer
    os.makedirs("checkpoints", exist_ok=True)
    tokenizer.save("checkpoints/smiles_tokenizer.json")

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
        vocab_size=len(tokenizer.vocab)
    ).to(device)

    # Freeze ESM parameters
    for param in model.protein_encoder.esm.parameters():
        param.requires_grad = False
    
    # # Enable gradient checkpointing for transformer decoder
    # model.decoder.enable_input_require_grads()
    # model.decoder.gradient_checkpointing_enable()
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop with memory optimizations
    global_step = 0
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Clear cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move data to device and immediately clear from CPU
            protein_data = {
                'coords': batch['protein']['coords'].to(device),
                'residue_indices': batch['protein']['residue_indices'].to(device)
            }
            smiles_tokens = batch['ligand']['smiles_tokens'].to(device)
            batch = None  # Clear the batch from CPU memory
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(protein_data, smiles_tokens[:, :-1])
                
                # Calculate loss (ignore padding tokens)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    smiles_tokens[:, 1:].contiguous().view(-1),
                    ignore_index=tokenizer.special_tokens['PAD']
                )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            
            # Free up memory
            del logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Log to wandb (less frequently)
            if global_step % 10 == 0:  # Log every 10 steps
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "global_step": global_step
                })
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

if __name__ == "__main__":
    train()