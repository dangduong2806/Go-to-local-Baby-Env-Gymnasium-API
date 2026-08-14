"""Evaluate a saved VLN Actor-Critic checkpoint."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from algorithms.rollout import observation_to_tensors
from envs.make_env import make_env
from models.actor_critic import ActorCritic


def get_action_name(env, action_id):
    """Return the readable MiniGrid action name for an action ID."""
    actions_enum = getattr(env.unwrapped, "actions", None)
    if actions_enum is None:
        return "unknown"

    try:
        return actions_enum(action_id).name
    except (TypeError, ValueError):
        return "unknown"


def annotate_frame(
    frame,
    mission,
    episode_number,
    episode_seed,
    step,
    action_text,
    reward,
    episode_return,
    status,
):
    """Add navigation context above an RGB environment frame."""
    frame_image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    header_height = 112
    canvas_width = max(560, frame_image.width)
    canvas = Image.new(
        "RGB",
        (canvas_width, frame_image.height + header_height),
        color=(18, 18, 18),
    )
    canvas.paste(
        frame_image,
        ((canvas_width - frame_image.width) // 2, header_height),
    )

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw.text(
        (10, 8),
        f"Episode: {episode_number}   Seed: {episode_seed}",
        fill="white",
        font=font,
    )
    draw.text(
        (10, 31),
        f"Mission: {mission}",
        fill=(255, 230, 80),
        font=font,
    )
    draw.text(
        (10, 54),
        f"Step: {step}   Action: {action_text}",
        fill="white",
        font=font,
    )
    draw.text(
        (10, 77),
        f"Reward: {reward:.3f}   Return: {episode_return:.3f}",
        fill="white",
        font=font,
    )

    status_color = {
        "SUCCESS": (80, 255, 120),
        "TRUNCATED": (255, 170, 70),
        "TERMINATED": (255, 100, 100),
    }.get(status, (180, 220, 255))
    draw.text(
        (390, 77),
        f"Status: {status}",
        fill=status_color,
        font=font,
    )

    return np.asarray(canvas)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/ppo_gotolocal/checkpoint_best.pt",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)

    # Now stochastic sampling will be the default.
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Use argmax actions instead of sampling from the trained "
            "policy distribution."
        ),
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
        help="Number of highest-return evaluation episodes to record.",
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

    number_of_actions = int(
        checkpoint["number_of_actions"]
    )

    if number_of_actions not in {3, 7}:
        raise ValueError(
            "Unsupported checkpoint action count: "
            f"{number_of_actions}. Expected 3 or 7."
        )

    navigation_actions_only = number_of_actions == 3

    model = ActorCritic(
        number_of_actions=number_of_actions,
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
        navigation_actions_only=navigation_actions_only,
    )

    episode_results = []
    best_video_candidates = []

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            observation, _ = env.reset(seed=episode_seed)

            # Use a fixed PyTorch seed for this episode so stochastic evaluation is
            # repeatable when the same checkpoint and episode seeds are used.
            torch.manual_seed(episode_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)

            mission = observation["mission"]
            
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            # We do not know whether an episode is one of the best until it
            # ends. Capture one episode at a time, then retain its frames only
            # if it belongs to the current top-N candidates.
            record_episode = (
                video_dir is not None
                and args.video_episodes > 0
            )
            video_frames = []

            if record_episode:
                # render_mode="rgb_array" returns the full environment as an
                # RGB frame. This is separate from the agent's partial RGB
                # observation used as the model input.
                video_frames.append(
                    annotate_frame(
                        frame=env.render(),
                        mission=mission,
                        episode_number=episode + 1,
                        episode_seed=episode_seed,
                        step=0,
                        action_text="reset",
                        reward=0.0,
                        episode_return=0.0,
                        status="START",
                    )
                )

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

                    if args.deterministic:
                        action = torch.argmax(logits, dim=-1)
                    else:
                        # PPO trains a stochastic categorical policy. Sampling here evaluates
                        # the same policy that was used during rollout collection.
                        distribution = torch.distributions.Categorical(
                            logits=logits
                        )
                        action = distribution.sample()

                action_id = int(action.item())
                observation, reward, terminated, truncated, _ = (
                    env.step(action_id)
                )
                episode_return += float(reward)
                episode_length += 1

                if record_episode:
                    if terminated:
                        status = (
                            "SUCCESS"
                            if episode_return > 0.0
                            else "TERMINATED"
                        )
                    elif truncated:
                        status = "TRUNCATED"
                    else:
                        status = "RUNNING"

                    video_frames.append(
                        annotate_frame(
                            frame=env.render(),
                            mission=mission,
                            episode_number=episode + 1,
                            episode_seed=episode_seed,
                            step=episode_length,
                            action_text=(
                                f"{get_action_name(env, action_id)} "
                                f"({action_id})"
                            ),
                            reward=float(reward),
                            episode_return=episode_return,
                            status=status,
                        )
                    )

            success = episode_return > 0.0
            episode_results.append(
                {
                    "episode": episode + 1,
                    "seed": episode_seed,
                    "mission": mission,
                    "return": episode_return,
                    "length": episode_length,
                    "success": success,
                    "video": None,
                }
            )

            if record_episode:
                best_video_candidates.append(
                    {
                        "episode_index": episode,
                        "episode_number": episode + 1,
                        "seed": episode_seed,
                        "return": episode_return,
                        "length": episode_length,
                        "frames": video_frames,
                    }
                )

                # Highest return is best. If returns are equal, prefer the
                # shorter episode, then the earlier episode number.
                best_video_candidates.sort(
                    key=lambda candidate: (
                        -candidate["return"],
                        candidate["length"],
                        candidate["episode_number"],
                    )
                )
                del best_video_candidates[args.video_episodes:]

            print(
                f"episode={episode + 1:3d} "
                f"return={episode_return:7.3f} "
                f"length={episode_length:4d} "
                f"success={success} "
                f"mission={mission!r}"
            )
    finally:
        env.close()

    # Encode only the selected top-N episodes after evaluation is complete.
    # This avoids writing a video for every episode while still allowing the
    # best episodes to be selected from the complete evaluation run.
    for rank, candidate in enumerate(best_video_candidates, start=1):
        video_path = video_dir / (
            f"rank_{rank:02d}_"
            f"episode_{candidate['episode_number']:03d}_"
            f"return_{candidate['return']:.3f}_"
            f"seed_{candidate['seed']}.mp4"
        )
        imageio.mimsave(
            video_path,
            candidate["frames"],
            fps=args.video_fps,
        )
        episode_results[candidate["episode_index"]]["video"] = str(
            video_path.resolve()
        )
        episode_results[candidate["episode_index"]]["video_rank"] = rank

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

        "policy": (
            "deterministic"
            if args.deterministic
            else "stochastic"
        ),

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
            f"saved the best {recorded_count} episode video(s) to: "
            f"{video_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
