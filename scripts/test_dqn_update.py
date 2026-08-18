"""Test one DQN optimization update.

Run from the project root:

    python -m scripts.test_dqn_update
"""

import math

import numpy as np
import torch

from algorithms.dqn import (
    synchronize_target_network,
    update_dqn,
)
from algorithms.replay_buffer import ReplayBuffer
from envs.make_env import make_env
from models.q_network import QNetwork


SEED = 42
TRANSITION_COUNT = 32
BATCH_SIZE = 8


def parameters_are_equal(
    first_model,
    second_model,
) -> bool:
    """Check whether two models have identical parameters."""

    return all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter
        in zip(
            first_model.parameters(),
            second_model.parameters(),
        )
    )


def parameters_changed(
    model,
    old_parameters,
) -> bool:
    """Check whether at least one model parameter changed."""

    return any(
        not torch.equal(parameter, old_parameter)
        for parameter, old_parameter
        in zip(
            model.parameters(),
            old_parameters,
        )
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = make_env(
        render_mode=None,
        rgb_partial_obs=True,
        navigation_actions_only=True,
        seed=SEED,
    )

    try:
        number_of_actions = env.action_space.n

        online_network = QNetwork(
            number_of_actions=number_of_actions,
        ).to(device)

        target_network = QNetwork(
            number_of_actions=number_of_actions,
        ).to(device)

        synchronize_target_network(
            online_network=online_network,
            target_network=target_network,
        )

        assert parameters_are_equal(
            online_network,
            target_network,
        ), (
            "The target network was not synchronized "
            "correctly."
        )

        assert all(
            not parameter.requires_grad
            for parameter in target_network.parameters()
        ), "Target-network parameters should be frozen."

        optimizer = torch.optim.Adam(
            online_network.parameters(),
            lr=1e-4,
        )

        replay_buffer = ReplayBuffer(
            capacity=TRANSITION_COUNT,
            seed=SEED,
        )

        observation, _ = env.reset(seed=SEED)

        for _ in range(TRANSITION_COUNT):
            action = env.action_space.sample()

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
            )

            if terminated or truncated:
                observation, _ = env.reset()
            else:
                observation = next_observation

        batch = replay_buffer.sample(
            batch_size=BATCH_SIZE,
            tokenizer=online_network.tokenizer,
            device=device,
        )

        old_online_parameters = [
            parameter.detach().clone()
            for parameter in online_network.parameters()
        ]

        old_target_parameters = [
            parameter.detach().clone()
            for parameter in target_network.parameters()
        ]

        result = update_dqn(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            batch=batch,
            gamma=0.99,
            max_gradient_norm=10.0,
            double_dqn=False,
        )

        print("DQN update")
        print("----------")
        print(f"loss: {result.loss:.6f}")
        print(
            f"mean selected Q-value: "
            f"{result.mean_q_value:.6f}"
        )
        print(
            f"mean target value: "
            f"{result.mean_target_value:.6f}"
        )
        print(
            f"mean absolute TD error: "
            f"{result.mean_absolute_td_error:.6f}"
        )
        print(
            f"gradient norm before clipping: "
            f"{result.gradient_norm:.6f}"
        )

        assert math.isfinite(result.loss)
        assert math.isfinite(result.mean_q_value)
        assert math.isfinite(result.mean_target_value)
        assert math.isfinite(
            result.mean_absolute_td_error
        )
        assert math.isfinite(result.gradient_norm)

        assert result.loss >= 0.0
        assert result.mean_absolute_td_error >= 0.0

        assert parameters_changed(
            online_network,
            old_online_parameters,
        ), (
            "The online-network parameters did not change."
        )

        # The optimizer must not update the target network.
        assert all(
            torch.equal(parameter, old_parameter)
            for parameter, old_parameter
            in zip(
                target_network.parameters(),
                old_target_parameters,
            )
        ), "The target-network parameters unexpectedly changed."

        # After synchronization, both networks should match again.
        synchronize_target_network(
            online_network=online_network,
            target_network=target_network,
        )

        assert parameters_are_equal(
            online_network,
            target_network,
        ), (
            "The networks do not match after "
            "target synchronization."
        )

        print("\nAll DQN-update checks passed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()