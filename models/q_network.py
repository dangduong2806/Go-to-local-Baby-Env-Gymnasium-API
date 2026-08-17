"""Vision-Language Q-network for DGN"""
from typing import Optional

import torch
from torch import nn

from .fusion import MultimodalFusion
from .language_encoder import BabyAITokenizer, LanguageEncoder
from .vision_encoder import VisionEncoder

class QNetwork(nn.Module):
    """Estimate one Q-value for every available action."""

    def __init__(
        self,
        number_of_actions: int,
        visual_dim: int = 128,
        language_dim: int = 128,
        fused_dim: int = 256,
    ):
        super().__init__()

        if number_of_actions <= 0:
            raise ValueError(
                "number_of_actions must be greater than zero."
            )

        self.number_of_actions = number_of_actions
        self.tokenizer = BabyAITokenizer()

        self.vision_encoder = VisionEncoder(
            embedding_dim=visual_dim,
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

        # One output for every discreate action:
        # Q(s, left), Q(s, right), Q(s, forward)
        self.q_head = nn.Linear(
            fused_dim,
            number_of_actions,
        )


    def forward(
        self, 
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
    ):
        """Estimate Q-values for a batch of observations.

        Args:
            images:
                RGB images with shape
                ``(batch, 3, height, width)``.

            token_ids:
                Tokenized missions with shape
                ``(batch, sequence_length)``.

            attention_mask:
                Mask with the same shape as ``token_ids``.
                Real tokens are 1 and padding tokens are 0.

            directions:
                Optional BabyAI directions with shape ``(batch,)``.
                When omitted, direction zero is used.

        Returns:
            Q-values with shape
            ``(batch, number_of_actions)``.
        """
        if images.ndim != 4:
            raise ValueError(
                "images must have shape "
                "(batch, channels, height, width)."
            )

        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape "
                "(batch, sequence_length)."
            )

        if images.shape[0] != token_ids.shape[0]:
            raise ValueError(
                "Image and token batch sizes must be equal."
            )

        if attention_mask.shape != token_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape "
                "as token_ids."
            )

        batch_size = images.shape[0]
        device = images.device

        if directions is None:
            directions = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )
        else:
            directions = directions.to(
                device=device,
                dtype=torch.long,
            ).reshape(-1)

        visual_embedding = self.vision_encoder(images)

        language_embedding = self.language_encoder(
            token_ids= token_ids,
            attention_mask=attention_mask,
        )

        fused_embedding = self.fusion(
            visual_embedding,
            language_embedding,
            directions,
        )

        q_values = self.q_head(fused_embedding)

        return q_values

    def select_greedy_action(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
    ):
        """Select the action with the highest predicted Q-value"""
        q_values = self(
            images=images,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=directions,
        )

        return q_values.argmax(dim=-1)