"""Experience replay buffer for DQN."""

from dataclasses import dataclass

import numpy as np
import torch

@dataclass
class ReplayBatch:
    """One sampled batch of DQN transitions"""
    images: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    directions: torch.Tensor

    actions: torch.Tensor
    rewards: torch.Tensor

    next_images: torch.Tensor
    next_token_ids: torch.Tensor
    next_attention_mask: torch.Tensor
    next_directions: torch.Tensor

    terminated: torch.Tensor
    truncated: torch.Tensor

@dataclass
class Transition:
    """One transition stored in the replay buffer"""
    image: np.ndarray
    mission: str
    direction: int

    action: int
    reward: float

    next_image: np.ndarray
    next_mission: str
    next_direction: int

    terminated: bool
    truncated: bool

class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer"""

    def __init__(
        self,
        capacity: int,
        seed: int | None = None,
    ):
        if capacity <= 0:
            raise ValueError(
                "Replay buffer capacity must be greater than zero,"
            )

        self.capacity = capacity
        self._storage: list[Transition] = []
        self._next_index = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self._storage)

    def add(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        terminated: bool,
        truncated: bool,
    ):
        """Add one environment transition"""
        image = np.asarray(
            observation["image"],
            dtype=np.uint8,
        )

        next_image = np.asarray(
            next_observation["image"],
            dtype=np.uint8,
        )

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                "Observation image must have shape "
                "(height, width, 3)."
            )

        if next_image.shape != image.shape:
            raise ValueError(
                "Current and next images must have equal shapes."
            )

        mission = observation["mission"]
        next_mission = next_observation["mission"]

        if not isinstance(mission, str) or not mission.strip():
            raise ValueError(
                "Observation mission must be a nonempty string."
            )

        if (
            not isinstance(next_mission, str)
            or not next_mission.strip()
        ):
            raise ValueError(
                "Next observation mission must be "
                "a nonempty string."
            )

        direction = int(observation.get("direction", 0))
        next_direction = int(
            next_observation.get("direction", 0)
        )

        if not 0 <= direction < 4:
            raise ValueError(
                "Observation direction must be between 0 and 3."
            )

        if not 0 <= next_direction < 4:
            raise ValueError(
                "Next observation direction must be "
                "between 0 and 3."
            )

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite.")

        transition = Transition(
            # Copy images so later environment operations cannot
            # mutate data already stored in the buffer.
            image=image.copy(),
            mission=mission,
            direction=direction,
            action=int(action),
            reward=float(reward),
            next_image=next_image.copy(),
            next_mission=next_mission,
            next_direction=next_direction,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._next_index] = transition

        self._next_index = (
            self._next_index + 1
        ) % self.capacity


    def sample(
        self,
        batch_size: int,
        tokenizer,
        device: torch.device,
    ):
        """Randomly sample transitions and convert them to tensors."""
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} transitions "
                f"from a buffer containing {len(self)}."
            )

        indices = self._rng.choice(
            len(self._storage),
            size=batch_size,
            replace=False,
        )

        transitions = [
            self._storage[int(index)]
            for index in indices
        ]

        images = torch.from_numpy(
            np.stack(
                [transition.image for transition in transitions]
            )
        ).permute(0, 3, 1, 2).contiguous().to(device)

        missions = [
            transition.mission
            for transition in transitions
        ]

        token_ids, attention_mask = tokenizer.encode_batch(
            missions=missions,
            device=device,
        )

        directions = torch.tensor(
            [
                transition.direction
                for transition in transitions
            ],
            dtype=torch.long,
            device=device,
        )

        actions = torch.tensor(
            [
                transition.action
                for transition in transitions
            ],
            dtype=torch.long,
            device=device
        )

        rewards = torch.tensor(
            [
                transition.reward
                for transition in transitions
            ],
            dtype=torch.float32,
            device=device,
        )

        next_images = torch.from_numpy(
            np.stack(
                [
                    transition.next_image
                    for transition in transitions
                ]
            )
        ).permute(0, 3, 1, 2).contiguous().to(device)

        next_missions = [
            transition.next_mission
            for transition in transitions
        ]

        (
            next_token_ids,
            next_attention_mask,
        ) = tokenizer.encode_batch(
            missions=next_missions,
            device=device,
        )

        next_directions = torch.tensor(
            [
                transition.next_direction
                for transition in transitions
            ],
            dtype=torch.long,
            device=device,
        )

        terminated = torch.tensor(
            [
                transition.terminated
                for transition in transitions
            ],
            dtype=torch.bool,
            device=device
        )

        truncated = torch.tensor(
            [
                transition.truncated
                for transition in transitions
            ],
            dtype=torch.bool,
            device=device,
        )

        return ReplayBatch(
            images=images,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=directions,
            actions=actions,
            rewards=rewards,
            next_images=next_images,
            next_token_ids=next_token_ids,
            next_attention_mask=next_attention_mask,
            next_directions=next_directions,
            terminated=terminated,
            truncated=truncated
        )

        