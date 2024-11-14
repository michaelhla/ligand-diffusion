import torch
import torch.nn as nn
from esm.models.esm3 import ESM3
from esm.utils.structure.affine3d import build_affine3d_from_coordinates

class ProteinEncoder(nn.Module):
    def __init__(self, window_size=512, window_overlap=32):
        """
        Args:
            window_size: Maximum number of residues per window (default 512 for ESM3)
            window_overlap: Number of residues to overlap between windows for smoother embeddings
        """
        super().__init__()
        self.esm = ESM3.from_pretrained()
        # Convert entire ESM model to bfloat16
        self.esm = self.esm.to(torch.bfloat16)
        self.max_seq_len = window_size
        self.window_overlap = window_overlap
        
    def _process_window(self, coords, residue_indices):
        """Process a single window of residues"""
        # Convert coordinates and indices to bfloat16
        coords = coords.to(torch.bfloat16)
        residue_indices = residue_indices.to(torch.bfloat16)
        
        # Calculate per-residue pLDDT from coordinate validity
        per_res_plddt = coords.isfinite().all(dim=-1).any(dim=-1).to(torch.bfloat16)
        
        # Forward pass through ESM3
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.esm(
                structure_coords=coords,
                per_res_plddt=per_res_plddt,
            )
        
        # Convert output back to float32 for the rest of our model
        return output.embeddings.to(torch.float32)

    def _get_windows(self, num_residues):
        """Generate window indices with overlap"""
        windows = []
        start = 0
        
        while start < num_residues:
            end = min(start + self.max_seq_len, num_residues)
            windows.append((start, end))
            
            # If we've reached the end, break
            if end == num_residues:
                break
                
            # Move start position, accounting for overlap
            start = end - self.window_overlap
            
        return windows
    
    def _merge_window_embeddings(self, window_embeddings, window_indices, num_residues, device):
        """
        Merge embeddings from multiple windows with linear interpolation in overlap regions
        """
        hidden_dim = window_embeddings[0].shape[-1]
        final_embeddings = torch.zeros((num_residues, hidden_dim), device=device)
        weight_sum = torch.zeros(num_residues, 1, device=device)
        
        for (start, end), emb in zip(window_indices, window_embeddings):
            # Create weight mask for linear interpolation
            weight_mask = torch.ones(end - start, 1, device=device)
            
            # Apply linear ramp in overlap regions
            if start > 0:  # Left overlap
                ramp_length = min(self.window_overlap, end - start)
                weight_mask[:ramp_length] = torch.linspace(0, 1, ramp_length, device=device).unsqueeze(1)
                
            if end < num_residues:  # Right overlap
                ramp_length = min(self.window_overlap, end - start)
                weight_mask[-ramp_length:] = torch.linspace(1, 0, ramp_length, device=device).unsqueeze(1)
            
            # Accumulate weighted embeddings and weights
            final_embeddings[start:end] += emb.squeeze(0) * weight_mask
            weight_sum[start:end] += weight_mask
        
        # Normalize by weight sum
        final_embeddings = final_embeddings / (weight_sum + 1e-8)
        
        return final_embeddings
        
    def forward(self, protein_data):
        """
        Args:
            protein_data: Dictionary containing:
                - coords: Tensor of shape (batch_size, num_residues, 3, 3) containing backbone coordinates
                - residue_indices: Tensor of shape (batch_size, num_residues) containing the original indices
                  of the residues in the protein sequence
        Returns:
            embeddings: Dictionary containing:
                - embeddings: Tensor of shape (batch_size, num_residues, hidden_dim)
                - residue_indices: Tensor of shape (batch_size, num_residues)
        """
        # Get device
        device = next(self.parameters()).device
        
        # Extract inputs
        coords = protein_data['coords'].to(device)  # (batch_size, num_residues, 3, 3)
        residue_indices = protein_data['residue_indices'].to(device)  # (batch_size, num_residues)
        
        batch_size, num_residues = coords.shape[:2]
        
        # Process each batch item separately
        all_embeddings = []
        
        for batch_idx in range(batch_size):
            if num_residues <= self.max_seq_len:
                # If within size limit, process directly
                embeddings = self._process_window(
                    coords[batch_idx:batch_idx+1],
                    residue_indices[batch_idx:batch_idx+1]
                )
                all_embeddings.append(embeddings.squeeze(0))
            else:
                # Get window indices
                windows = self._get_windows(num_residues)
                
                # Process each window
                window_embeddings = []
                for start, end in windows:
                    window_coords = coords[batch_idx:batch_idx+1, start:end]
                    window_indices = residue_indices[batch_idx:batch_idx+1, start:end]
                    emb = self._process_window(window_coords, window_indices)
                    window_embeddings.append(emb)
                
                # Merge window embeddings
                merged_embeddings = self._merge_window_embeddings(
                    window_embeddings,
                    windows,
                    num_residues,
                    device
                )
                all_embeddings.append(merged_embeddings)
        
        # Stack batch embeddings
        final_embeddings = torch.stack(all_embeddings)
        
        return {
            'embeddings': final_embeddings,  # (batch_size, num_residues, hidden_dim)
            'residue_indices': residue_indices  # (batch_size, num_residues)
        }