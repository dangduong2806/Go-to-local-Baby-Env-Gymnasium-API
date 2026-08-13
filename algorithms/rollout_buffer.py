"""On-policy rollout storage for PPO."""
from dataclasses import dataclass

import torch

from torch.nn.utils.rnn import pad_sequence

@dataclass
class RolloutBatch:
    """Tensor representation of one complete rollout."""
    images: torch.Tensor
    token_ids: torch.Tensor
    attention_masks: torch.Tensor
    directions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor

class RolloutBuffer:
    """Store a fixed number of transitions collected from the old policy.
    GAE is intentionally not calculated here. The buffer only stores the
    observations and policy outputs required later by GAE and PPO.
    """
    def __init__(
            self,
            capacity: int,
            pad_token_id: int,
    ):
        self.capacity = capacity
        self.pad_token_id = pad_token_id
        self.clear()

    def clear(self):
        self.images = []
        self.token_ids = []
        self.attention_masks = []
        self.directions = []
        self.actions = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.old_log_probs = []
        self.values = []
        self.next_values = []

    def __len__(self) -> int:
        return len(self.actions)

    def add(
         self,
        image: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        direction: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
        old_log_prob: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
    ):
        """Add one transition without retaining an autograd graph."""
        if len(self) >= self.capacity:
            raise RuntimeError("Rollout buffer is already full.")

        # Remove the batch dimension because collection uses batch size one.
        # Store data on CPU to avoid filling accelerator memory during rollout.
        self.images.append(image.squeeze(0).detach().cpu())
        self.token_ids.append(token_ids.squeeze(0).detach().cpu())
        self.attention_masks.append(
            attention_mask.squeeze(0).detach().cpu()
        )
        self.directions.append(
            direction.reshape(-1)[0].detach().cpu()
        )
        self.actions.append(action.reshape(-1)[0].detach().cpu())
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.old_log_probs.append(
            old_log_prob.reshape(-1)[0].detach().cpu()
        )
        self.values.append(value.reshape(-1)[0].detach().cpu())
        self.next_values.append(
            next_value.reshape(-1)[0].detach().cpu()
        )

    def as_tensors(
        self,
        device: torch.device | None = None,
    ):
        """Stack the rollout into tensors suitable for GAE and PPO."""
        if len(self) == 0:
            raise RuntimeError("Cannot tensorize an empty rollout.")

        batch = RolloutBatch(
            images=torch.stack(self.images),
            token_ids=pad_sequence(
                self.token_ids,
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            attention_masks=pad_sequence(
                self.attention_masks,
                batch_first=True,
                padding_value=0,
            ),
            directions=torch.stack(self.directions).long(),
            actions=torch.stack(self.actions).long(),
            rewards=torch.tensor(self.rewards, dtype=torch.float32),
            terminated=torch.tensor(
                self.terminated,
                dtype=torch.bool,
            ),
            truncated=torch.tensor(
                self.truncated,
                dtype=torch.bool,
            ),
            old_log_probs=torch.stack(self.old_log_probs).float(),
            values=torch.stack(self.values).float(),
            next_values=torch.stack(self.next_values).float(),
        )

        if device is None:
            return batch

        return RolloutBatch(
            **{
                name: tensor.to(device)
                for name, tensor in vars(batch).items()
            }
        )
