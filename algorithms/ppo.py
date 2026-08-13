"""Explicit PyTorch implementation of PPO."""
from dataclasses import dataclass

import torch
from torch import nn

from .rollout_buffer import RolloutBuffer


@dataclass
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    ratio_before_update: float


class PPOTrainer:
    """Perform PPO updates on fixed on-policy rollout data."""
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    def update(
        self,
        rollout: RolloutBuffer,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        debug: bool = False,
    ):
        """Apply multiple PPO epochs to one rollout.
        PPO is on-policy. The stored old log probabilities remain fixed for
        every epoch, and the rollout is discarded after this update.
        """
        batch = rollout.as_tensors(device=self.device)

        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        if advantages.shape != batch.rewards.shape:
            raise ValueError("Advantages have an incorrect shape.")

        if returns.shape != batch.rewards.shape:
            raise ValueError("Returns have an incorrect shape.")

        if not torch.isfinite(advantages).all():
            raise ValueError("Advantages contain non-finite values.")

        if not torch.isfinite(returns).all():
            raise ValueError("Returns contain non-finite values.")

        # Advantage normalization normally improves optimization stability by
        # keeping policy-gradient magnitudes on a predictable scale.
        normalized_advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

        self.model.train()

        # Before the first optimizer step, the rollout policy and current
        # policy are identical, so the probability ratio should be near 1.
        with torch.no_grad():
            _, initial_log_prob, _, _ = (
                self.model.get_action_and_value(
                    images=batch.images,
                    token_ids=batch.token_ids,
                    attention_mask=batch.attention_masks,
                    directions=batch.directions,
                    action=batch.actions,
                )
            )

            initial_ratio = torch.exp(
                initial_log_prob - batch.old_log_probs
            )

            ratio_before_update = initial_ratio.mean().item()

        if debug:
            print(
                "mean PPO ratio before first update: "
                f"{ratio_before_update:.6f}"
            )

        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []
        approximate_kls = []
        clip_fractions = []

        batch_size = len(rollout)

        for _ in range(self.update_epochs):
            shuffled_indices = torch.randperm(
                batch_size,
                device=self.device,
            )
            for start in range(0, batch_size, self.minibatch_size,):
                minibatch_indices = shuffled_indices[
                    start : start + self.minibatch_size
                ]

                # Evaluate the actions actually taken during rollout.
                # Do not sample replacement actions here
                _, new_log_prob, entropy, new_value = (
                    self.model.get_action_and_value(
                        images=batch.images[minibatch_indices],
                        token_ids=batch.token_ids[minibatch_indices],
                        attention_mask=(
                            batch.attention_masks[minibatch_indices]
                        ),
                        directions=batch.directions[minibatch_indices],
                        action=batch.actions[minibatch_indices],
                    )
                )

                old_log_prob = batch.old_log_probs[
                    minibatch_indices
                ]

                minibatch_advantages = normalized_advantages[
                    minibatch_indices
                ]

                minibatch_returns = returns[minibatch_indices]

                log_ratio = new_log_prob - old_log_prob
                ratio = torch.exp(log_ratio)

                unclipped_objective = (
                    ratio * minibatch_advantages
                )

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                )

                clipped_objective = (
                    clipped_ratio * minibatch_advantages
                )

                # PyTorch minimizes, so negate PPO's clipped objective.
                policy_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                # The critic regresses toward GAE-based return targets.
                value_loss = 0.5 * (
                    (new_value - minibatch_returns) ** 2
                ).mean()

                mean_entropy = entropy.mean()

                # Policy and value terms are minimized. Entropy is subtracted
                # because higher entropy encourages continued exploration.
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * mean_entropy
                )

                for name, loss_value in {
                    "policy loss": policy_loss,
                    "value loss": value_loss,
                    "total loss": total_loss,
                }.items():
                    if not torch.isfinite(loss_value):
                        raise ValueError(
                            f"{name} became non-finite."
                        )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean()

                    clip_fraction = (
                        torch.abs(ratio - 1.0)
                        > self.clip_coef
                    ).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(mean_entropy.item())
                total_losses.append(total_loss.item())
                approximate_kls.append(approximate_kl.item())
                clip_fractions.append(clip_fraction.item())

        # Calculate explained variance using the updated value function.
        with torch.no_grad():
            _, updated_values = self.model(
                images=batch.images,
                token_ids=batch.token_ids,
                attention_mask=batch.attention_masks,
                directions=batch.directions,
            )

            return_variance = torch.var(
                returns,
                unbiased=False,
            )

            if return_variance.item() < 1e-8:
                explained_variance = 0.0
            else:
                explained_variance = (
                    1.0
                    - torch.var(
                        returns - updated_values,
                        unbiased=False,
                    )
                    / return_variance
                ).item()

        def mean(values: list[float]) -> float:
            return sum(values) / len(values)

        return PPOMetrics(
            policy_loss=mean(policy_losses),
            value_loss=mean(value_losses),
            entropy=mean(entropies),
            total_loss=mean(total_losses),
            approx_kl=mean(approximate_kls),
            clip_fraction=mean(clip_fractions),
            explained_variance=explained_variance,
            ratio_before_update=ratio_before_update,
        )
