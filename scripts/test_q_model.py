"""Inspect and validate the vision-language Q-network.
Run from the project root:

    python -m scripts.test_q_model
"""
import numpy as np
import torch
from torch import nn

from algorithms.rollout import observation_to_tensors
from envs.make_env import make_env
from models.q_network import QNetwork

SEED = 42
TEST_BATCH_SIZE = 4

def module_has_nonzero_gradient(module: nn.Module) -> bool:
    """Return whether a module received at least one nonzero gradient."""
    for parameter in module.parameters():
        if (
            parameter.grad is not None
            and torch.any(parameter.grad != 0)
        ):
            return True

    return False

def print_gradient_result(
    name: str,
    module: nn.Module,
) -> bool:
    """Print and return a module's gradient test result."""
    received_gradient = module_has_nonzero_gradient(module)

    print(
        f"{name} received nonzero gradients: "
        f"{received_gradient}"
    )

    return received_gradient

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
        observation, _ = env.reset(seed=SEED)

        number_of_actions = env.action_space.n

        assert number_of_actions == 3, (
            "The navigation-only environment should expose "
            "exactly three actions."
        )

        model = QNetwork(
            number_of_actions=number_of_actions,
        ).to(device)

        model.train()

        (
            image,
            token_ids,
            attention_mask,
            direction,
        ) = observation_to_tensors(
            observation=observation,
            tokenizer=model.tokenizer,
            device=device,
        )

        print("Environment and input")
        print("---------------------")
        print(f"device: {device}")
        print(f"mission: {observation['mission']}")
        print(f"image shape: {tuple(image.shape)}")
        print(f"token IDs shape: {tuple(token_ids.shape)}")

        print(
            "attention mask shape: "
            f"{tuple(attention_mask.shape)}"
        )

        print(f"direction shape: {tuple(direction.shape)}")
        print(f"number of actions: {number_of_actions}")

        # Test a single observation
        q_values = model(
            images=image,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=direction,
        )

        print("\nSingle-observation forward pass")
        print("-------------------------------")
        print(f"Q-values: {q_values.detach()}")
        print(f"Q-value shape: {tuple(q_values.shape)}")

        assert q_values.shape == (1, number_of_actions), (
            "Unexpected single-observation Q-value shape."
        )

        assert torch.isfinite(q_values).all(), (
            "The Q-network produced NaN or infinite values."
        )

        # Test greedy action selection
        greedy_action = model.select_greedy_action(
            images=image,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=direction,
        )

        expected_action = q_values.argmax(dim=-1)

        print("\nGreedy action")
        print("-------------")

        print(f"selected action: {greedy_action.item()}")
        print(
            "largest-Q action: "
            f"{expected_action.item()}"
        )

        assert greedy_action.shape == (1,), (
            "The greedy action should have shape (1,)."
        )

        assert torch.equal(
            greedy_action,
            expected_action,
        ), (
            "select_greedy_action did not select "
            "the largest Q-value."
        )

        assert 0 <= greedy_action.item() < number_of_actions, (
            "The greedy action is outside the action space."
        )

        # Repeat the observation to test batched inputs.
        batch_images = image.repeat(
            TEST_BATCH_SIZE,
            1,
            1,
            1,
        )

        batch_token_ids = token_ids.repeat(
            TEST_BATCH_SIZE,
            1,
        )

        batch_attention_mask = attention_mask.repeat(
            TEST_BATCH_SIZE,
            1,
        )

        batch_directions = direction.repeat(
            TEST_BATCH_SIZE,
        )

        batch_q_values = model(
            images=batch_images,
            token_ids=batch_token_ids,
            attention_mask=batch_attention_mask,
            directions=batch_directions,
        )

        batch_actions = model.select_greedy_action(
            images=batch_images,
            token_ids=batch_token_ids,
            attention_mask=batch_attention_mask,
            directions=batch_directions,
        )

        print("\nBatch forward pass")
        print("------------------")

        print(
            "batch Q-value shape: "
            f"{tuple(batch_q_values.shape)}"
        )
        print(
            "batch action shape: "
            f"{tuple(batch_actions.shape)}"
        )

        assert batch_q_values.shape == (
            TEST_BATCH_SIZE,
            number_of_actions,
        ), "Unexpected batch Q-value shape."

        assert batch_actions.shape == (
            TEST_BATCH_SIZE,
        ), "Unexpected batch action shape."

        assert torch.isfinite(batch_q_values).all(), (
            "The batched Q-values contain NaN or infinity."
        )

        assert torch.all(
            (batch_actions >= 0)
            & (batch_actions < number_of_actions)
        ), "A batched action is outside the action space."

        # Simulate a Q-learning loss and check gradient flow
        actions = torch.tensor(
            [0, 1, 2, 0],
            dtype=torch.long,
            device=device,
        )

        selected_q_values = batch_q_values.gather(
            dim=1,
            index=actions.unsqueeze(1),
        ).squeeze(1)

        # Artificial targets are sufficient for checking backpropagation.
        target_q_values = torch.tensor(
            [1.0, 0.5, -0.25, 0.75],
            dtype=torch.float32,
            device=device,
        )

        loss = torch.nn.functional.smooth_l1_loss(
            selected_q_values,
            target_q_values,
        )

        model.zero_grad()
        loss.backward()

        print("\nGradient inspection")
        print("-------------------")
        print(f"artificial DQN loss: {loss.item():.6f}")

        gradient_results = {
            "vision encoder": print_gradient_result(
                "vision encoder",
                model.vision_encoder,
            ),
            "language encoder": print_gradient_result(
                "language encoder",
                model.language_encoder,
            ),
            "fusion module": print_gradient_result(
                "fusion module",
                model.fusion,
            ),
            "Q-value head": print_gradient_result(
                "Q-value head",
                model.q_head,
            ),
        }

        assert all(gradient_results.values()), (
            "At least one Q-network component did not receive "
            "a nonzero gradient."
        )

        print("\nAll Q-network checks passed.")

    finally:
        env.close()

if __name__ == "__main__":
    main()