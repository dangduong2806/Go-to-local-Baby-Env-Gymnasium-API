"""Simplified vision-language actor-critic network."""
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

        # Actor: answers "What should I do?"
        # It produces one unnormalized logit per discrete action.
        self.actor_head = nn.Linear(
            fused_dim,
            number_of_actions,
        )

        # Critic: answers "How good is my current situation?"
        # It estimates expected future discounted return, not immediate reward.
        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, 1),
        )

    def forward(
            self,
            images: torch.Tensor,
            token_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            directions: Optional[torch.Tensor] = None,
    ):
        """Run a complete vision-language forward pass.
        Args:
            images:
                RGB tensor with shape ``(batch, 3, height, width)``.

            token_ids:
                Token IDs with shape ``(batch, sequence_length)``.

            attention_mask:
                Mask with the same shape as ``token_ids``. Real tokens
                are 1 and padding tokens are 0.

            directions:
                Optional BabyAI directions with shape ``(batch,)``. When
                omitted, direction zero is used.

        Returns:
            logits:
                Tensor with shape ``(batch, number_of_actions)``.

            values:
                Tensor with shape ``(batch,)``.
        """
        if images.shape[0] != token_ids.shape[0]:
            raise ValueError(
                "Image and token batch sizes must be equal."
            )

        if attention_mask.shape != token_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as token_ids."
            )
        device = images.device

        if directions is None:
            directions = torch.zeros(
                images.shape[0],
                dtype=torch.long,
                device=device
            ) 

        visual_embedding = self.vision_encoder(images)

        language_embedding = self.language_encoder(
            token_ids = token_ids,
            attention_mask = attention_mask
        )

        fused_embedding = self.fusion(
            visual_embedding,
            language_embedding,
            directions,
        ) 

        logits = self.actor_head(fused_embedding)
        values = self.critic_head(fused_embedding).squeeze(-1)

        return logits, values


    def get_action_distribution(
            self,
            images,
            token_ids,
            attention_mask,
            directions=None,
    ):
        logits, _ = self(
            images=images,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=directions,
        )

        return torch.distributions.Categorical(logits=logits)


    def sample_action(
            self,
            images,
            token_ids,
            attention_mask,
            directions=None,
    ):
        distribution = self.get_action_distribution(
            images=images,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=directions,
        )

        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, log_prob, entropy


    def get_action_and_value(
            self,
            images,
            token_ids,
            attention_mask,
            directions=None,
            action=None,
    ):
        logits, value = self(
            images=images,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=directions,
        )

        distribution = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = distribution.sample()
        else:
            action = action.to(
                device=logits.device,
                dtype=torch.long,
            ).reshape(-1)

            if action.shape[0] != logits.shape[0]:
                raise ValueError(
                    "Action batch size must match observation batch size."
                )

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, log_prob, entropy, value