"""Test the DQN experience replay buffer.

Run from the project root:

    python -m scripts.test_replay_buffer
"""

import numpy as np
import torch

from algorithms.replay_buffer import ReplayBuffer
from envs.make_env import make_env
from models.q_network import QNetwork


SEED = 42
CAPACITY = 5
TRANSITIONS_TO_ADD = 8
BATCH_SIZE = 4


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
        model = QNetwork(
            number_of_actions=env.action_space.n,
        ).to(device)

        replay_buffer = ReplayBuffer(
            capacity=CAPACITY,
            seed=SEED,
        )

        observation, _ = env.reset(seed=SEED)

        for step in range(TRANSITIONS_TO_ADD):
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

            print(
                f"step={step + 1}, "
                f"buffer size={len(replay_buffer)}"
            )

            if terminated or truncated:
                observation, _ = env.reset()
            else:
                observation = next_observation

        # We added eight transitions to a capacity-five buffer.
        # The oldest transitions should have been overwritten.
        assert len(replay_buffer) == CAPACITY

        batch = replay_buffer.sample(
            batch_size=BATCH_SIZE,
            tokenizer=model.tokenizer,
            device=device,
        )

        image_shape = observation["image"].shape
        height, width, channels = image_shape

        print("\nSampled batch")
        print("-------------")
        print(f"images: {tuple(batch.images.shape)}")
        print(f"token IDs: {tuple(batch.token_ids.shape)}")
        print(
            "attention mask: "
            f"{tuple(batch.attention_mask.shape)}"
        )
        print(
            f"directions: {tuple(batch.directions.shape)}"
        )
        print(f"actions: {tuple(batch.actions.shape)}")
        print(f"rewards: {tuple(batch.rewards.shape)}")
        print(
            f"next images: {tuple(batch.next_images.shape)}"
        )
        print(
            f"terminated: {tuple(batch.terminated.shape)}"
        )
        print(
            f"truncated: {tuple(batch.truncated.shape)}"
        )

        assert batch.images.shape == (
            BATCH_SIZE,
            channels,
            height,
            width,
        )

        assert batch.next_images.shape == (
            BATCH_SIZE,
            channels,
            height,
            width,
        )

        assert batch.token_ids.shape[0] == BATCH_SIZE
        assert batch.attention_mask.shape == batch.token_ids.shape

        assert batch.next_token_ids.shape[0] == BATCH_SIZE
        assert (
            batch.next_attention_mask.shape
            == batch.next_token_ids.shape
        )

        assert batch.directions.shape == (BATCH_SIZE,)
        assert batch.actions.shape == (BATCH_SIZE,)
        assert batch.rewards.shape == (BATCH_SIZE,)
        assert batch.next_directions.shape == (BATCH_SIZE,)
        assert batch.terminated.shape == (BATCH_SIZE,)
        assert batch.truncated.shape == (BATCH_SIZE,)

        assert batch.images.dtype == torch.uint8
        assert batch.next_images.dtype == torch.uint8
        assert batch.actions.dtype == torch.long
        assert batch.rewards.dtype == torch.float32
        assert batch.terminated.dtype == torch.bool
        assert batch.truncated.dtype == torch.bool

        assert torch.all(
            (batch.actions >= 0)
            & (batch.actions < env.action_space.n)
        )

        assert torch.isfinite(batch.rewards).all()

        # Verify that a sampled batch can pass through QNetwork.
        q_values = model(
            images=batch.images,
            token_ids=batch.token_ids,
            attention_mask=batch.attention_mask,
            directions=batch.directions,
        )

        next_q_values = model(
            images=batch.next_images,
            token_ids=batch.next_token_ids,
            attention_mask=batch.next_attention_mask,
            directions=batch.next_directions,
        )

        assert q_values.shape == (
            BATCH_SIZE,
            env.action_space.n,
        )

        assert next_q_values.shape == (
            BATCH_SIZE,
            env.action_space.n,
        )

        print(
            f"current Q-values: {tuple(q_values.shape)}"
        )
        print(
            f"next Q-values: {tuple(next_q_values.shape)}"
        )
        print("\nAll replay-buffer checks passed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()