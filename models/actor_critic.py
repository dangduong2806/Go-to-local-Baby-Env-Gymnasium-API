"""Simplified vision-language actor-critic network."""

from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn

from .fusion import MultimodalFusion
from .language_encoder import BabyAITokenizer, LanguageEncoder
from .vision_encoder import VisionEncoder

class ActorCritic(nn.Module):
    """Produce action logits and a state-value estimate from a VLN observation."""
    def __init__(
            self,
            number_of_actions: int,
            visual_dim: int = 128,
            language_dim: int = 128,
            fused_dim: int = 256,
    ):
        super().__init__()

        self.tokenizer = BabyAITokenizer()

        self.vision_encoder = VisionEncoder(
            embedding_dim=visual_dim
        )

        self.language_encoder = LanguageEncoder(
            vocabulary_size=len(self.tokenizer),
            pad_token_id=self.tokenizer.pad_token_id,
            word_embedding_dim=64,
            hidden_dim=language_dim,
        )

        self.fusion = MultimodalFusion(
            visual_dim=visual_dim,
            language_dim=language_dim,
            direction_dim=16,
            fused_dim=fused_dim,
        )

        # The policy head produces one unnormalized logit per discrete action.
        self.policy_head = nn.Linear(
            fused_dim,
            number_of_actions,
        )   
        # This is only a model component in Milestone 2. It is not trained.
        self.value_head = nn.Linear(
            fused_dim,
            1,
        )

    def encode_language(
            self, 
            missions: Sequence[str],
            device: torch.device,
    ):
        """Tokenize and encode a batch of mission strings."""
        token_ids, lengths = self.tokenizer.encode_batch(
            missions=missions,
            device=device,
        )     

        return self.language_encoder(token_ids, lengths)

    def forward(
            self,
            images: torch.Tensor,
            missions: Sequence[str],
            directions: Optional[torch.Tensor] = None,
    ):
        """Run a complete vision-language forward pass.
        Args:
            images:
                RGB tensor with shape ``(batch, 3, height, width)``.

            missions:
                One mission string per image.

            directions:
                Optional BabyAI directions with shape ``(batch,)``. When
                omitted, direction zero is used.

        Returns:
            logits:
                Tensor with shape ``(batch, number_of_actions)``.

            values:
                Tensor with shape ``(batch,)``.
        """
        if images.shape[0] != len(missions):
            raise ValueError(
                "The image batch size must equal the number of missions."
            )
        device = images.device

        if directions is None:
            directions = torch.zeros(
                images.shape[0],
                dtype=torch.long,
                device=device
            ) 

        visual_embedding = self.vision_encoder(images)
        language_embedding = self.encode_language(missions=missions, device=device)

        fused_embedding = self.fusion(
            visual_embedding,
            language_embedding,
            directions,
        ) 

        logits = self.policy_head(fused_embedding)
        values = self.value_head(fused_embedding).squeeze(-1)

        return logits, values