"""Fusion layer for visual, language, and direction features."""
import torch
from torch import nn

class MultimodalFusion(nn.Module):
    """Fuse vision, language, and agent-direction embeddings."""
    def __init__(
            self, 
            visual_dim: int = 128,
            language_dim: int = 128,
            direction_dim: int = 16,
            fused_dim: int = 256,
    ):
        super().__init__()

        # BabyAI direction is one of four discrete orientations.
        self.direction_embedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=direction_dim,
        )      

        self.fusion_layer = nn.Sequential(
            nn.Linear(
                visual_dim + language_dim + direction_dim,
                fused_dim,
            ),
            nn.ReLU(),
        )

    def forward(
            self,
            visual_embedding: torch.Tensor,
            language_embedding: torch.Tensor,
            direction: torch.Tensor
    ):
        """Combine the three observation components."""
        if direction.ndim != 1:
            direction = direction.reshape(-1)

        direction = direction.to(
            device=visual_embedding.device,
            dtype=torch.long,
        )

        direction_embedding = self.direction_embedding(direction)

        combined = torch.cat(
            [
                visual_embedding,
                language_embedding,
                direction_embedding,
            ],
            dim=-1
        )

        return self.fusion_layer(combined)
      
        
