"""Optimization utilities for Deep Q-Learning."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .replay_buffer import ReplayBatch

@dataclass
class DQNUpdateResult:
    """Statistics produced by one DQN optimization step."""
    loss: float
    mean_q_value: float
    mean_target_value: float
    mean_absolute_td_error: float
    gradient_norm: float

@torch.no_grad()
def synchronize_target_network(
    online_network,
    target_network,
):
    """Copy online-network parameters into the target network."""
    target_network.load_state_dict(
        online_network.state_dict()
    )

    target_network.eval()

    # The target network only creates fixed Q-learning targets.
    # It must never receive optimizer updates.
    target_network.requires_grad_(False)

def update_dqn(
    online_network,
    target_network,
    optimizer,
    batch: ReplayBatch,
    gamma: float = 0.99,
    max_gradient_norm: float = 10.0,
    double_dqn: bool = False,
):
    """Perform one DQN optimization step.
    Args:
        online_network:
            Q-network updated by the optimizer.

        target_network:
            Frozen network used to calculate stable targets.

        optimizer:
            Optimizer containing only online-network parameters.

        batch:
            Random minibatch sampled from the replay buffer.

        gamma:
            Discount factor for future rewards.

        max_gradient_norm:
            Maximum gradient norm used for clipping.

        double_dqn:
            If False, use the standard DQN target.
            If True, use the Double-DQN target.

    Returns:
        Scalar statistics describing this update.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must be between zero and one."
        )

    if max_gradient_norm <= 0.0:
        raise ValueError(
            "max_gradient_norm must be greater than zero."
        )

    online_network.train()
    target_network.eval()

    # Estimate Q(s, a) for every action in each current state.
    all_q_values = online_network(
        images=batch.images,
        token_ids=batch.token_ids,
        attention_mask=batch.attention_mask,
        directions=batch.directions
    )

    number_of_actions = all_q_values.shape[1]

    if torch.any(batch.actions < 0):
        raise ValueError(
            "The batch contains a negative action ID."
        )
    if torch.any(batch.actions >= number_of_actions):
        raise ValueError(
            "The batch contains an action outside "
            "the network's action space."
        )

    # Keep only the Q-value for the action that was actually taken.
    #
    # all_q_values:
    #     (batch, number_of_actions)
    #
    # selected_q_values:
    #     (batch,)
    selected_q_values = all_q_values.gather(
        dim=1,
        index=batch.actions.unsqueeze(1),
    ).squeeze(1)

    with torch.no_grad():
        if double_dqn:
            # The online network chooses the next action.
            online_next_q_values = online_network(
                images=batch.next_images,
                token_ids=batch.next_token_ids,
                attention_mask=batch.next_attention_mask,
                directions=batch.next_directions,
            )

            next_actions = online_next_q_values.argmax(
                dim=1,
                keepdim=True,
            )

            # The target network evaluates the action selected
            # by the online network.
            target_next_q_values = target_network(
                images=batch.next_images,
                token_ids=batch.next_token_ids,
                attention_mask=batch.next_attention_mask,
                directions=batch.next_directions,
            )

            next_q_values = target_next_q_values.gather(
                dim=1,
                index=next_actions,
            ).squeeze(1)
        else:
            # Standard DQN both selects and evaluates the
            # next action using the target network.
            target_next_q_values = target_network(
                images=batch.next_images,
                token_ids=batch.next_token_ids,
                attention_mask=batch.next_attention_mask,
                directions=batch.next_directions,
            )

            next_q_values = target_next_q_values.max(
                dim=1
            ).values

        # A true terminal state has no future value.
        # A time-limit truncation is not necessarily terminal,
        # so truncated is intentionally not used in this mask.
        bootstrap_mask = (
            ~batch.terminated
        ).to(dtype=torch.float32)

        target_q_values = (
            batch.rewards
            + gamma * bootstrap_mask * next_q_values
        )

    temporal_difference_errors = (
        target_q_values - selected_q_values
    )

    loss = F.smooth_l1_loss(
        selected_q_values,
        target_q_values,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        online_network.parameters(),
        max_norm=max_gradient_norm,
    )

    optimizer.step()

    return DQNUpdateResult(
        loss=float(loss.detach().item()),
        mean_q_value=float(
            selected_q_values.detach().mean().item()
        ),
        mean_target_value=float(
            target_q_values.detach().mean().item()
        ),
        mean_absolute_td_error=float(
            temporal_difference_errors
            .detach()
            .abs()
            .mean()
            .item()
        ),
        gradient_norm=float(
            gradient_norm.detach().item()
        ),
    )