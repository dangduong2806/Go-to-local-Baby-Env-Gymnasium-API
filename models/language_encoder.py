"""Tokenizer and GRU encoder for BabyAI language instructions."""
import re
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class BabyAITokenizer:
    """A small word-level tokenizer for BabyAI GoToLocal missions.
    The vocabulary includes the words normally used by GoToLocal missions.
    Unknown words map to the ``<unk>`` token.
    """
    def __init__(self):
        words = [
            "<pad>",
            "<unk>",
            "go",
            "to",
            "the",
            "a",
            "red",
            "green",
            "blue",
            "purple",
            "yellow",
            "grey",
            "gray",
            "ball",
            "box",
            "key",
        ]

        self.token_to_id = {
            word: index for index, word in enumerate(words)
        }

        self.pad_token_id = self.token_to_id["<pad>"]
        self.unk_token_id = self.token_to_id["<unk>"]

    def __len__(self):
        return len(self.token_to_id)

    def tokenize(self, mission: str):
        # Lowercase a mission and split it into words
        return re.findall(r"[a-z]+", mission.lower())

    def encode(self, mission: str):
        # Convert one mission into token IDs.
        tokens = self.tokenize(mission=mission)

        if not tokens:
            return [self.unk_token_id]

        return [
            self.token_to_id.get(token, self.unk_token_id)
            for token in tokens
        ]

    def encode_batch(
            self,
            missions: Sequence[str],
            device: torch.device
    ):
        # Pad and encode a batch of mission strings
        """
        Returns:
            token_ids:
                Long tensor with shape ``(batch, maximum_length)``.

            attention_mask:
                Long tensor with shape ``(batch, maximum_length)``.
                Real tokens are 1 and padding tokens are 0.
        """
        encoded = [self.encode(mission=mission) for mission in missions]

        if not encoded:
            raise ValueError("At least one mission is required.")

        maximum_length = max(len(sequence) for sequence in encoded)

        token_ids = torch.full(
            (len(encoded), maximum_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.zeros(
            (len(encoded), maximum_length),
            dtype=torch.long,
            device=device,
        )

        for row, sequence in enumerate(encoded):
            sequence_length = len(sequence)

            token_ids[row, :sequence_length] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )

            attention_mask[row, :sequence_length] = 1

        return token_ids, attention_mask

class LanguageEncoder(nn.Module):
    """Encode BabyAI mission strings using an embedding layer and GRU."""
    def __init__(
            self, 
            vocabulary_size: int,
            pad_token_id: int, 
            word_embedding_dim: int = 64,
            hidden_dim: int = 128,
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=word_embedding_dim,
            padding_idx=pad_token_id,
        )

        self.gru = nn.GRU(
            input_size=word_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(
            self,
            token_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ):
        """Encode padded token sequences.
        Args:
            token_ids:
                Tensor with shape ``(batch, sequence_length)``.

            attention_mask:
                Tensor with the same shape as ``token_ids``. Real tokens
                are 1 and padding tokens are 0.

        Returns:
            Tensor with shape ``(batch, hidden_dim)``.
        """
        if attention_mask.shape != token_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as token_ids."
            )

        lengths = attention_mask.sum(dim=1).to(dtype=torch.long)

        if torch.any(lengths <= 0):
            raise ValueError(
                "Every instruction must contain at least one token."
            )
        
        embedded_words = self.word_embedding(token_ids)
        # Packing prevents padding tokens from influencing the final GRU state.
        packed_words = pack_padded_sequence(
            embedded_words,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )       

        _, final_hidden_state = self.gru(packed_words)

        # GRU output shape is (layers, batch, hidden_dim). We use the final
        # state from the only GRU layer
        return final_hidden_state[-1]
       
       
