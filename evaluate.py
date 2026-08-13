"""Evaluate a saved VLN Actor-Critic checkpoint."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from algorithms.rollout import observation_to_tensors
from envs.make_env import make_env
from models.actor_critic import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/ppo_gotolocal/checkpoint_best.pt",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of selecting the largest logit.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Display the environment during evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ppo_gotolocal/evaluation_results.json",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Optional directory for MP4 evaluation videos.",
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes to record.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Playback speed for saved evaluation videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    if args.video_episodes < 0:
        raise ValueError("--video-episodes cannot be negative.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be greater than zero.")
    if args.render and args.video_dir is not None:
        raise ValueError(
            "Use either --render or --video-dir, not both. Gymnasium "
            "selects one render mode when the environment is created."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    # Only load checkpoints created locally by this project. PyTorch
    # checkpoints use pickle and should not be loaded from untrusted sources.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = ActorCritic(
        number_of_actions=checkpoint["number_of_actions"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.video_dir is not None:
        render_mode = "rgb_array"
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
    else:
        render_mode = "human" if args.render else None
        video_dir = None

    env = make_env(
        render_mode=render_mode,
        rgb_partial_obs=True,
    )

    episode_results = []

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            observation, _ = env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            video_path = None

            record_episode = (
                video_dir is not None
                and episode >= args.episodes - args.video_episodes
            )
            video_frames = []

            if record_episode:
                # render_mode="rgb_array" returns the full environment as an
                # RGB frame. This is separate from the agent's partial RGB
                # observation used as the model input.
                video_frames.append(env.render())

            while not terminated and not truncated:
                image, token_ids, attention_mask, direction = (
                    observation_to_tensors(
                        observation=observation,
                        tokenizer=model.tokenizer,
                        device=device,
                    )
                )

                with torch.no_grad():
                    logits, _ = model(
                        images=image,
                        token_ids=token_ids,
                        attention_mask=attention_mask,
                        directions=direction,
                    )

                    if args.stochastic:
                        distribution = torch.distributions.Categorical(
                            logits=logits
                        )
                        action = distribution.sample()
                    else:
                        action = torch.argmax(logits, dim=-1)

                action_id = int(action.item())
                observation, reward, terminated, truncated, _ = (
                    env.step(action_id)
                )
                episode_return += float(reward)
                episode_length += 1

                if record_episode:
                    video_frames.append(env.render())

            if record_episode:
                video_path = video_dir / (
                    f"episode_{episode + 1:03d}_"
                    f"seed_{episode_seed}.mp4"
                )
                imageio.mimsave(
                    video_path,
                    video_frames,
                    fps=args.video_fps,
                )

            success = episode_return > 0.0
            episode_results.append(
                {
                    "episode": episode + 1,
                    "seed": episode_seed,
                    "return": episode_return,
                    "length": episode_length,
                    "success": success,
                    "video": (
                        str(video_path.resolve())
                        if video_path is not None
                        else None
                    ),
                }
            )

            print(
                f"episode={episode + 1:3d} "
                f"return={episode_return:7.3f} "
                f"length={episode_length:4d} "
                f"success={success}"
            )
    finally:
        env.close()

    returns = [result["return"] for result in episode_results]
    lengths = [result["length"] for result in episode_results]
    successes = [
        float(result["success"]) for result in episode_results
    ]

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_update": checkpoint["update"],
        "checkpoint_environment_steps": checkpoint[
            "environment_steps"
        ],
        "evaluation_episodes": args.episodes,
        "policy": "stochastic" if args.stochastic else "deterministic",
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "video_directory": (
            str(video_dir.resolve()) if video_dir is not None else None
        ),
    }

    output = {
        "summary": summary,
        "episodes": episode_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\nEvaluation summary")
    print("------------------")
    print(f"mean return: {summary['mean_return']:.4f}")
    print(
        "mean episode length: "
        f"{summary['mean_episode_length']:.2f}"
    )
    print(f"success rate: {summary['success_rate']:.2%}")
    print(f"results saved to: {output_path.resolve()}")
    if video_dir is not None:
        recorded_count = min(args.episodes, args.video_episodes)
        print(
            f"saved {recorded_count} video(s) to: "
            f"{video_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
