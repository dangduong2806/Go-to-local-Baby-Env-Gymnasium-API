"""Inspect BabyAI observations, actions, and episode transitions.
Run without visualization:

    python -m envs.inspect_env

Run with a human-rendered window:

    python -m envs.inspect_env --render
"""
import argparse
from typing import Any

import numpy as np

from .make_env import ENV_ID, make_env

FIXED_SEED = 42
NUM_RANDOM_STEPS = 10

def get_action_name(env: Any, action_id: int):
    """Return the MiniGrid enum name associated with an action ID."""
    actions_enum = getattr(env.unwrapped, "actions", None)

    if actions_enum is None:
        return "unknown"

    try:
        return actions_enum(action_id).name
    except (TypeError, ValueError):
        return "unknown"

def print_action_meanings(env: Any):
    """Print all actions using the environment's MiniGrid action enum."""
    print("\nDiscrete action meanings:")

    actions_enum = getattr(env.unwrapped, "actions", None)

    if actions_enum is None:
        print("  MiniGrid action enum is not available.")
        return

    for action in actions_enum:
        print(f" {action.value}: {action.name}")

def print_initial_observation(env: Any, obs: dict[str, Any]):
    """Print the important parts of the initial VLN observation."""
    image = obs["image"]

    env_id = env.spec.id if env.spec is not None else ENV_ID

    print("Environment inspection")
    print("----------------------")
    print(f"environment ID: {env_id}")
    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")
    print(f"observation dictionary keys: {list(obs.keys())}")
    print(f"image shape: {image.shape}")
    print(f"image dtype: {image.dtype}")
    print(f"minimum pixel value: {np.min(image)}")
    print(f"maximum pixel value: {np.max(image)}")
    print(f"mission: {obs['mission']}")

    if "direction" in obs:
        print(f"direction: {obs['direction']}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect BabyAI-GoToLocal-v0 using random actions."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Display the environment in a human-rendered window.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    render_mode = "human" if args.render else None

    env = make_env(
        render_mode=render_mode,
        rgb_partial_obs=True,
        seed=FIXED_SEED,
    )

    total_reward = 0.0
    steps_executed = 0
    terminated = False
    truncated = False

    try:
        # Reset explicitly to obtain the initial observation and info.
        obs, info = env.reset(seed = FIXED_SEED)

        # Seed random action sampling so repeated runs use the same actions.
        env.action_space.seed(FIXED_SEED)

        print_initial_observation(env, obs)
        print_action_meanings(env=env)

        if args.render:
            env.render()

        print("\nRandom episode transitions")
        print("--------------------------")

        for timestep in range(1, NUM_RANDOM_STEPS + 1):
            action_id = int(env.action_space.sample())
            action_name = get_action_name(env, action_id)

            obs, reward, terminated, truncated, info = env.step(action_id)

            total_reward += float(reward)
            steps_executed += 1

            print(f"\ntimestep: {timestep}")
            print(f"selected action ID: {action_id}")
            print(f"selected action name: {action_name}")
            print(f"reward: {reward}")
            print(f"terminated: {terminated}")
            print(f"current mission: {obs['mission']}")
            print(f"new image shape: {obs['image'].shape}")

            if "direction" in obs:
                print(f'new direction: {obs["direction"]}')

            if args.render:
                env.render()

            if terminated or truncated:
                print("\nEpisode ended early.")
                break

        print("\nEpisode summary")
        print("---------------")
        print(f"steps executed: {steps_executed}")
        print(f"total reward: {total_reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")

        if not terminated and not truncated:
            print(
                f"episode still active after "
                f"{NUM_RANDOM_STEPS} inspection steps"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    
    
