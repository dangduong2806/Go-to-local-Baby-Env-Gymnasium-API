"""Generalized Advantage Estimation."""
import torch

def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
):
    """Calculate GAE advantages and value-function returns.
    The temporal-difference residual is:

        delta_t = reward_t + gamma * V(s_{t+1}) - V(s_t)

    GAE accumulates these residuals backward through an episode. Lambda
    controls the bias-variance tradeoff: smaller values use more one-step
    bootstrapping, while values near one use longer return estimates.

    A true termination receives no next-state bootstrap. A truncation may use
    the value of its final observation, but the recursive advantage is stopped
    so it cannot leak into the reset episode that follows.
    """
    expected_shape = rewards.shape

    for name, tensor in {
        "values": values,
        "next_values": next_values,
        "terminated": terminated,
        "truncated": truncated,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"but received {tensor.shape}."
            )

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(
        (),
        dtype=rewards.dtype,
        device=rewards.device
    )

    for timestep in reversed(range(len(rewards))):
        # Bootstrap through ordinary transitions and pure truncations, but
        # never through a true MDP termination.
        bootstrap_mask = (~terminated[timestep]).float()

        td_residual = (
            rewards[timestep]
            + gamma
            * next_values[timestep]
            * bootstrap_mask
            - values[timestep]
        )

        # Both termination and truncation mark an episode boundary. Stopping
        # this recursive term prevents advantage leakage into a reset episode.
        episode_continues = ~(
            terminated[timestep] | truncated[timestep]
        )

        last_advantage = (
            td_residual
            + gamma
            * gae_lambda
            * episode_continues.float()
            * last_advantage
        )

        advantages[timestep] = last_advantage

    returns = advantages + values

    # advantages → train Actor
    # returns    → train Critic

    return advantages, returns
