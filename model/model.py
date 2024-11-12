import torch
import torch.nn as nn
from model.esm import ProteinEncoder
from model.diffusion import DiscreteDiffusion

class ProteinLigandDiffusion(nn.Module):
    def __init__(self, hidden_dim, num_steps, vocab_size):
        super().__init__()
        self.protein_encoder = ProteinEncoder()
        self.diffusion = DiscreteDiffusion(num_steps, vocab_size)
        
        # Transformer decoder that conditions on protein structure
        self.decoder = nn.TransformerDecoder(
            num_layers=6,
            d_model=hidden_dim,
            nhead=8
        )
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedding = nn.Embedding(num_steps, hidden_dim)
        
    def forward(self, protein_data, ligand_tokens, t):
        # Get protein structure embeddings
        protein_embed = self.protein_encoder(protein_data)
        
        # Mask embeddings to only use interface residues
        interface_mask = protein_data['interface_mask']
        protein_embed = protein_embed * interface_mask.unsqueeze(-1)
        
        # Get noisy tokens at timestep t
        noisy_tokens = self.diffusion.q_sample(ligand_tokens, t)
        
        # Embed noisy tokens and timestep
        token_embed = self.token_embedding(noisy_tokens)
        time_embed = self.time_embedding(t)
        
        # Combine embeddings
        decoder_input = token_embed + time_embed
        
        # Predict clean tokens
        pred_logits = self.decoder(
            decoder_input,
            protein_embed,
            tgt_mask=self.generate_square_subsequent_mask(ligand_tokens.size(1))
        )
        
        return pred_logits
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generate causal mask for transformer"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask