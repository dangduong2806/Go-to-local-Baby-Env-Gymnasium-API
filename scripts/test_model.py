"""Inspect the Milestone 2 vision-language model.
Run from the project root:

    python -m scripts.test_model
"""
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from envs.make_env import make_env
from models.actor_critic import ActorCritic

SEED = 42

def image_to_tensor(image: np.ndarray):
    """Convert one HWC NumPy RGB image into a batched BCHW tensor."""
    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )

def module_has_nonzero_gradient(module: nn.Module) -> bool:
    """Return True if at least one module parameter has a nonzero gradient."""
    for parameter in module.parameters():
        if parameter.grad is not None and torch.any(parameter.grad != 0):
            return True

    return False

def print_gradient_result(name: str, module: nn.Module) -> bool:
    received_gradient = module_has_nonzero_gradient(module)
    print(f"{name} received nonzero gradients: {received_gradient}")
    return received_gradient

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = make_env(
        render_mode=None,
        rgb_partial_obs=True,
        seed=SEED,
    )

    try:
        obs_1, _ = env.reset(seed=SEED)

        image_1 = image_to_tensor(obs_1["image"])
        mission = obs_1["mission"]
        direction_1 = torch.tensor(
            [obs_1.get("direction", 0)],
            dtype=torch.long,
        )

        # Turning left changes the agent-centered partial visual observation
        # without requiring a successful policy or any training.
        left_action = int(env.unwrapped.actions.left)
        obs_2, _, terminated, truncated, _ = env.step(left_action)

        if terminated or truncated:
            raise RuntimeError(
                "The inspection episode unexpectedly ended after one turn."
            )

        image_2 = image_to_tensor(obs_2["image"])

        number_of_actions = env.action_space.n 
        model = ActorCritic(number_of_actions=number_of_actions)
        token_ids, attention_mask = model.tokenizer.encode_batch(
            missions=[mission],
            device=image_1.device,
        )

        print("Model")
        print("-----")
        print(model)

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )

        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print(f"\ntotal parameters: {parameter_count:,}")
        print(f"trainable parameters: {trainable_parameter_count:,}")

        # Forward-pass shape
        logits, values = model(
            images=image_1,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=direction_1,
        )

        probabilities = torch.softmax(logits, dim=-1)
        probability_sum = probabilities.sum(dim=-1)

        action, log_prob, entropy, value = model.get_action_and_value(
            images=image_1,
            token_ids=token_ids,
            attention_mask=attention_mask,
            directions=direction_1,
        )

        print(f"sampled action: {action}")
        print(f"log probability: {log_prob}")
        print(f"entropy: {entropy}")
        print(f"critic value: {value}")

        print("\nForward-pass inspection")
        print("-----------------------")
        print(f"input image shape: {tuple(image_1.shape)}")
        print(f"mission: {mission}")
        print(f"logits shape: {tuple(logits.shape)}")
        print(f"value shape: {tuple(values.shape)}")
        print(f"action probabilities: {probabilities.detach()}")
        print(f"probability sum: {probability_sum.item():.6f}")

        assert torch.allclose(
            probability_sum,
            torch.ones_like(probability_sum),
            atol=1e-6,
        ), "Action probabilities do not sum to one."

    finally:
        env.close()

if __name__ == "__main__":
    main()