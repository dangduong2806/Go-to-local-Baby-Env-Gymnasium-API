"""Collect on-policy experience from a BabyAI environment."""
from dataclasses import dataclass, field

import numpy as np
import torch

from .rollout_buffer import RolloutBuffer

@dataclass
class CollectorState:
    """State retained between consecutive rollout collections."""

    observation: dict | None = None
    episode_return: float = 0.0
    episode_length: int = 0

@dataclass
class RolloutStatistics:
    """Statistics for episodes completed during one rollout."""

    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_successes: list[float] = field(default_factory=list)

def observation_to_tensors(
    observation: dict,
    tokenizer,
    device: torch.device,
):
    """Convert one BabyAI observation into batched model inputs."""
    image = (
        torch.from_numpy(np.asarray(observation["image"]))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device)
    )

    token_ids, attention_mask = tokenizer.encode_batch(
        missions=[observation["mission"]],
        device=device,
    )

    direction = torch.tensor(
        [observation.get("direction", 0)],
        dtype=torch.long,
        device=device,
    )

    return image, token_ids, attention_mask, direction


def evaluate_value(
        model,
        observation: dict,
        tokenizer,
        device: torch.device,
):
    """Evaluate V(s) without constructing a gradient graph."""
    image, token_ids, attention_mask, direction = (
        observation_to_tensors(
            observation=observation,
            tokenizer=tokenizer,
            device=device
        )
    )

    _, value = model(
        images=image,
        token_ids=token_ids,
        attention_mask=attention_mask,
        directions=direction,
    )

    return value

def collect_rollout(
        env,
        model,
        tokenizer,
        rollout_steps: int,
        device: torch.device,
        collector_state: CollectorState | None = None,
):
    """Collect exactly ``rollout_steps`` on-policy transitions.
    ``collector_state`` preserves unfinished episodes across rollout
    boundaries. This avoids resetting an episode merely because a rollout
    buffer became full.
    """
    if collector_state is None:
        collector_state = CollectorState()

    if collector_state.observation is None:
        collector_state.observation, _ = env.reset()

    statistics = RolloutStatistics()

    buffer = RolloutBuffer(
        capacity=rollout_steps,
        pad_token_id=tokenizer.pad_token_id,
    )

    model.eval()

    for _ in range(rollout_steps):
        observation = collector_state.observation

        image, token_ids, attention_mask, direction = (
            observation_to_tensors(
                observation,
                tokenizer,
                device,
            )
        )
        # Rollout collection does not train the model and must not retain
        # autograd graphs.
        with torch.no_grad():
            action, log_prob, _, value = (
                model.get_action_and_value(
                    images=image,
                    token_ids=token_ids,
                    attention_mask=attention_mask,
                    directions=direction,
                )
            )
        action_id = int(action.item())

        (
            next_observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action_id)

        # Gymnasium distinguishes two types of episode endings:
        #
        # terminated: a real MDP terminal state, such as success.
        # truncated: an external cutoff, usually a time limit.
        #
        # True termination has no future value. A pure truncation can still
        # bootstrap from the valid final observation.

        if terminated:
            next_value = torch.zeros(
                1,
                dtype=torch.float32,
                device=device,
            )
        else:
            with torch.no_grad():
                next_value = evaluate_value(
                    model,
                    next_observation,
                    tokenizer,
                    device,
                )

        buffer.add(
            image=image,
            token_ids=token_ids,
            attention_mask=attention_mask,
            direction=direction,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            old_log_prob=log_prob,
            value=value,
            next_value=next_value,
        )

        collector_state.episode_return += float(reward)
        collector_state.episode_length += 1

        if terminated or truncated:
            statistics.episode_returns.append(
                collector_state.episode_return
            )
            statistics.episode_lengths.append(
                collector_state.episode_length
            )

            # BabyAI gives positive reward on successful task completion.
            statistics.episode_successes.append(
                float(
                    terminated
                    and collector_state.episode_return > 0.0
                )
            )

            collector_state.observation, _ = env.reset()
            collector_state.episode_return = 0.0
            collector_state.episode_length = 0
        else:
            collector_state.observation = next_observation

    assert len(buffer) == rollout_steps

    return buffer, collector_state, statistics
         


