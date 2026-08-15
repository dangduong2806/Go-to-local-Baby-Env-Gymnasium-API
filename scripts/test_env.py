"""Check raw environment observations and model-ready inputs."""
import numpy as np
import torch

from algorithms.rollout import observation_to_tensors
from envs.make_env import make_env
from models.actor_critic import ActorCritic

import matplotlib.pyplot as plt
from pathlib import Path

def check_observation(observation):
    # Validate one raw observation returned by the environment
    assert isinstance(observation, dict)

    image = observation["image"]
    mission = observation["mission"]
    direction = observation["direction"]

    # Raw image
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[-1] == 3
    assert image.dtype == np.uint8
    assert image.min() >= 0
    assert image.max() <= 255

    # Language instruction
    assert isinstance(mission, str)
    assert mission.strip()

    # BabyAI has 4 possible orientations: 0, 1, 2, 3
    assert isinstance(direction, (int, np.integer))
    assert 0 <= direction < 4

def main():
    env = make_env(
        render_mode=None,
        rgb_partial_obs=True,
        navigation_actions_only=True,
    )

    try:
        observation, info = env.reset(seed=42)

        check_observation(observation=observation)

        print("Complete observation:")
        print(observation)

        print("\nImage:")
        print(observation["image"])
        print("Shape:", observation["image"].shape)
        print("Data type:", observation["image"].dtype)

        print("\nMission:")
        print(observation["mission"])

        print("\nAdditional info:")
        print(info)

        print("\nAction space:")
        print(env.action_space)

        image = observation["image"]
        mission = observation["mission"]
        direction = observation["direction"]

        output_path = Path("images/observation.png")

        plt.figure(figsize=(6, 6))
        plt.imshow(image, interpolation="nearest")
        plt.title(
            f"Model's partial view\n"
            f"Mission: {mission} | Direction: {direction}"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        print("Saved image to:", output_path.resolve())

    finally:
        env.close()

if __name__ == "__main__":
    main()