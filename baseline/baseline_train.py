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
from datasets.embedding_dataloader import EmbeddingDataset, collate_embeddings
from baseline.baseline_model import BaselineModel
from datasets.pdbbind import PDBBind
from datasets.moad import MOAD
from model.smiles_tokenizer import SMILESBPETokenizer
import json

login(token=open("api_keys.txt").readlines()[0].split("hf: ")[1].strip())
wandb.login(key=open("api_keys.txt").readlines()[1].split("wandb: ")[1].strip())


def train():
    wandb.init(
        project="ligand-diffusion",
        name="baseline-embeddings",
        config={
            "learning_rate": 1e-4,
            "batch_size": 64,  # Can use larger batches now
            "max_epochs": 100,
            "hidden_dim": 1024,
            "num_layers": 6,
            "num_heads": 8,
            "grad_clip": 1.0,
        }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = SMILESBPETokenizer(vocab_size=1000)
    tokenizer.load("checkpoints/smiles_tokenizer.json")
    
    # Create dataset from H5 files
    dataset = EmbeddingDataset(
        h5_paths=[
            "/workspace/embeddings/pdbbind_train_embeddings.h5",
            "/workspace/embeddings/moad_train_embeddings.h5"
        ],
        smiles_tokenizer=tokenizer
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_embeddings
    )
    
    # Initialize model
    model = BaselineModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        vocab_size=len(tokenizer.vocab)
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    global_step = 0
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move data to device
            protein_embeddings = batch['protein_embeddings'].to(device)
            smiles_tokens = batch['smiles_tokens'].to(device)
            
            # Forward pass
            logits = model(protein_embeddings, smiles_tokens[:, :-1])
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                smiles_tokens[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.special_tokens['PAD']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Log to wandb
            if global_step % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "global_step": global_step
                })
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader),
            }, f"checkpoints/baseline_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()