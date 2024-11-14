import torch
import torch.nn as nn
from model.esm import ProteinEncoder

class BaselineModel(nn.Module):
    def __init__(self, hidden_dim=1024, num_layers=6, num_heads=8, vocab_size=1000):
        super().__init__()
        self.protein_encoder = ProteinEncoder()
        
        # Add projection layer from ESM embedding dim (1536) to our hidden dim
        self.protein_proj = nn.Linear(1536, hidden_dim)
        
        # Transformer decoder with gradient checkpointing
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            batch_first=True,
            dropout=0.1  # Add some dropout
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize with smaller values
        nn.init.normal_(self.pos_embedding, std=0.02)
        
    def forward(self, protein_data, smiles_tokens):
        """
        Args:
            protein_data: Dictionary containing:
                - coords: Tensor of shape (batch_size, num_residues, 3, 3) containing backbone coordinates
                - residue_indices: Tensor of shape (batch_size, num_residues) containing residue indices
            smiles_tokens: Tensor of shape (batch_size, seq_len) containing tokenized SMILES
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Get protein embeddings from ESM
        protein_encoder_output = self.protein_encoder(protein_data)
        protein_embed = protein_encoder_output['embeddings']  # (batch_size, num_residues, 1536)
        
        # Project to hidden_dim
        protein_embed = self.protein_proj(protein_embed)  # (batch_size, num_residues, hidden_dim)
        
        # Optional: Use interface mask if available
        if 'interface_mask' in protein_data:
            interface_mask = protein_data['interface_mask'].unsqueeze(-1)
            protein_embed = protein_embed * interface_mask
            
        # Embed SMILES tokens and add positional encoding
        token_embed = self.token_embedding(smiles_tokens)
        seq_len = token_embed.size(1)
        token_embed = token_embed + self.pos_embedding[:, :seq_len, :]
        
        # Create causal mask for autoregressive generation
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(token_embed.device)
        
        # Decode SMILES tokens conditioned on protein embedding
        decoder_output = self.decoder(
            token_embed,
            protein_embed,
            tgt_mask=causal_mask
        )
        
        # Project to vocabulary size
        logits = self.output_layer(decoder_output)
        
        return logits
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generate causal mask for transformer"""
        # Add size check to prevent CUDA errors
        if sz <= 0:
            raise ValueError(f"Invalid sequence length for mask: {sz}")
            
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask
        
    def generate(self, protein_data, max_length=512, temperature=1.0, tokenizer=None):
        """
        Autoregressive generation of SMILES string
        
        Args:
            protein_data: Dictionary containing protein structure information
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            tokenizer: SMILESBPETokenizer instance for start/end token IDs
        Returns:
            generated_tokens: Tensor of shape (batch_size, seq_len) containing generated tokens
        """
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for generation")
            
        device = next(self.parameters()).device
        batch_size = protein_data['coords'].size(0)
        
        # Start with start token
        curr_tokens = torch.full(
            (batch_size, 1), 
            fill_value=tokenizer.special_tokens['START'],
            device=device
        )
        
        # Generate tokens auto-regressively
        for _ in range(max_length-1):
            logits = self.forward(protein_data, curr_tokens)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(next_token_logits.softmax(dim=-1), num_samples=1)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            
            # Stop if all sequences have generated END token
            if (curr_tokens == tokenizer.special_tokens['END']).any(dim=1).all():
                break
                
        return curr_tokens