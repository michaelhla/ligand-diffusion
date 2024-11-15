import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, hidden_dim=1024, num_layers=6, num_heads=8, vocab_size=1000):
        super().__init__()
        
        # Remove ESM encoder and just keep projection layer
        self.protein_proj = nn.Linear(1536, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            batch_first=True,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        nn.init.normal_(self.pos_embedding, std=0.02)
        
    def forward(self, protein_embeddings, smiles_tokens):
        """
        Args:
            protein_embeddings: Tensor of shape (batch_size, num_residues, 1536) containing ESM embeddings
            smiles_tokens: Tensor of shape (batch_size, seq_len) containing tokenized SMILES
        """
        # Project protein embeddings
        protein_embed = self.protein_proj(protein_embeddings)
        
        # Embed SMILES tokens and add positional encoding
        token_embed = self.token_embedding(smiles_tokens)
        seq_len = token_embed.size(1)
        token_embed = token_embed + self.pos_embedding[:, :seq_len, :]
        
        # Create causal mask
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(token_embed.device)
        
        # Decode SMILES tokens
        decoder_output = self.decoder(
            token_embed,
            protein_embed,
            tgt_mask=causal_mask
        )
        
        return self.output_layer(decoder_output)
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        if sz <= 0:
            raise ValueError(f"Invalid sequence length for mask: {sz}")
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask