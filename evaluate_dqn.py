"""Evaluate a trained DQN or Double-DQN checkpoint.

Basic evaluation:

    python evaluate_dqn.py \
        --checkpoint runs/dqn_gotolocal/checkpoint_best.pt \
        --episodes 100

Evaluate and save videos:

    python evaluate_dqn.py \
        --checkpoint runs/dqn_gotolocal/checkpoint_best.pt \
        --episodes 100 \
        --video-dir runs/dqn_gotolocal/videos \
        --video-episodes 3
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from algorithms.rollout import observation_to_tensors
from envs.make_env import make_env
from models.q_network import QNetwork

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/dqn_gotolocal/checkpoint_best.pt",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1_000,
        help="Seed of the first evaluation episode.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "runs/dqn_gotolocal/"
            "evaluation_results.json"
        ),
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Display the environment in a window.",
    )

    parser.add_argument(
        "--show-steps",
        action="store_true",
        help=(
            "Print Q-values and the selected action "
            "at every environment step."
        ),
    )

    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Optional directory for evaluation MP4 videos.",
    )

    parser.add_argument(
        "--video-episodes",
        type=int,
        default=3,
        help=(
            "Number of highest-return episodes to save "
            "as videos."
        ),
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Playback speed of saved videos.",
    )

    return parser.parse_args()

def validate_args(args):
    """Validate evaluation arguments."""

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than zero."
        )

    if args.video_episodes < 0:
        raise ValueError(
            "--video-episodes cannot be negative."
        )

    if args.video_fps <= 0:
        raise ValueError(
            "--video-fps must be greater than zero."
        )

    if args.render and args.video_dir is not None:
        raise ValueError(
            "Use either --render or --video-dir, not both. "
            "The environment selects one render mode when "
            "it is created."
        )

def get_action_name(environment, action_id):
    """Return a readable action name."""

    actions_enum = getattr(
        environment.unwrapped,
        "actions",
        None,
    )

    if actions_enum is None:
        return "unknown"

    try:
        return actions_enum(action_id).name
    except (TypeError, ValueError):
        return "unknown"

def format_q_values(q_values, action_names):
    """Create a readable Q-value string."""

    parts = []

    for action_id, q_value in enumerate(q_values):
        action_name = action_names.get(
            action_id,
            f"action_{action_id}",
        )

        parts.append(
            f"{action_name}={q_value:.3f}"
        )

    return " | ".join(parts)

def annotate_frame(
    frame,
    mission,
    episode_number,
    episode_seed,
    step,
    action_text,
    q_value_text,
    reward,
    episode_return,
    status,   
):
    """Add DQN information above an RGB frame."""

    frame_image = Image.fromarray(
        np.asarray(frame, dtype=np.uint8)
    )

    header_height = 145
    canvas_width = max(680, frame_image.width)

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            frame_image.height + header_height,
        ),
        color=(18, 18, 18),
    )

    canvas.paste(
        frame_image,
        (
            (canvas_width - frame_image.width) // 2,
            header_height,
        ),
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            16,
        )
    except OSError:
        font = ImageFont.load_default()

    draw.text(
        (10, 8),
        (
            f"Episode: {episode_number}   "
            f"Seed: {episode_seed}"
        ),
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
        f"Q-values: {q_value_text}",
        fill=(120, 210, 255),
        font=font,
    )

    draw.text(
        (10, 100),
        (
            f"Reward: {reward:.3f}   "
            f"Return: {episode_return:.3f}"
        ),
        fill="white",
        font=font,
    )

    status_color = {
        "SUCCESS": (80, 255, 120),
        "TRUNCATED": (255, 170, 70),
        "TERMINATED": (255, 100, 100),
        "RUNNING": (180, 220, 255),
        "START": (180, 220, 255),
    }.get(
        status,
        (255, 255, 255),
    )

    draw.text(
        (500, 100),
        f"Status: {status}",
        fill=status_color,
        font=font,
    )

    return np.asarray(canvas)

def load_checkpoint(
    checkpoint_path,
    device,  
):
    """Load and validate a DQN checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    # PyTorch checkpoints may contain pickle data.
    # Only load checkpoints created by this project.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "algorithm",
        "online_model_state_dict",
        "number_of_actions",
        "environment_steps",
    }

    missing_keys = required_keys.difference(
        checkpoint.keys()
    )

    if missing_keys:
        raise KeyError(
            "The checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    algorithm = checkpoint["algorithm"]

    if algorithm not in {"dqn", "double_dqn"}:
        raise ValueError(
            "This evaluator requires a DQN checkpoint. "
            f"Found algorithm={algorithm!r}."
        )

    number_of_actions = int(
        checkpoint["number_of_actions"]
    )

    if number_of_actions not in {3, 7}:
        raise ValueError(
            "Unsupported checkpoint action count: "
            f"{number_of_actions}. Expected 3 or 7."
        )

    model = QNetwork(
        number_of_actions=number_of_actions,
    ).to(device)

    model.load_state_dict(
        checkpoint["online_model_state_dict"]
    )

    model.eval()

    return checkpoint, model

def evaluate_episode(
    model,
    environment,
    episode_number,
    episode_seed,
    device,
    record_video,
    show_steps,
):
    """Run one greedy DQN evaluation episode."""

    observation, _ = environment.reset(
        seed=episode_seed
    )

    mission = observation["mission"]

    terminated = False
    truncated = False
    episode_return = 0.0
    episode_length = 0

    selected_q_values = []
    maximum_q_values = []
    action_counts = {
        action_id: 0
        for action_id in range(
            model.number_of_actions
        )
    }

    action_names = {
        action_id: get_action_name(
            environment,
            action_id,
        )
        for action_id in range(
            model.number_of_actions
        )
    }

    frames = []

    if record_video:
        frames.append(
            annotate_frame(
                frame=environment.render(),
                mission=mission,
                episode_number=episode_number,
                episode_seed=episode_seed,
                step=0,
                action_text="reset",
                q_value_text="not evaluated",
                reward=0.0,
                episode_return=0.0,
                status="START",
            )
        )

    while not terminated and not truncated:
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

        with torch.no_grad():
            q_values_tensor = model(
                images=image,
                token_ids=token_ids,
                attention_mask=attention_mask,
                directions=direction,
            )

            # Evaluation uses the greedy DQN policy.
            action_tensor = q_values_tensor.argmax(
                dim=1
            )

        action_id = int(action_tensor.item())

        q_values = (
            q_values_tensor[0]
            .detach()
            .cpu()
            .tolist()
        )

        chosen_q_value = float(
            q_values[action_id]
        )

        maximum_q_value = float(
            max(q_values)
        )

        action_name = action_names[action_id]

        selected_q_values.append(chosen_q_value)
        maximum_q_values.append(maximum_q_value)
        action_counts[action_id] += 1

        (
            next_observation,
            reward,
            terminated,
            truncated,
            _,
        ) = environment.step(action_id)

        episode_return += float(reward)
        episode_length += 1

        q_value_text = format_q_values(
            q_values=q_values,
            action_names=action_names,
        )

        if show_steps:
            print(
                f"  step={episode_length:3d} "
                f"action={action_name}({action_id}) "
                f"chosen_Q={chosen_q_value:9.4f} "
                f"reward={float(reward):7.3f} "
                f"return={episode_return:7.3f}"
            )

            print(
                f"    {q_value_text}"
            )

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

        if record_video:
            frames.append(
                annotate_frame(
                    frame=environment.render(),
                    mission=mission,
                    episode_number=episode_number,
                    episode_seed=episode_seed,
                    step=episode_length,
                    action_text=(
                        f"{action_name} ({action_id})"
                    ),
                    q_value_text=q_value_text,
                    reward=float(reward),
                    episode_return=episode_return,
                    status=status,
                )
            )

        observation = next_observation

    success = episode_return > 0.0

    readable_action_counts = {
        action_names[action_id]: count
        for action_id, count
        in action_counts.items()
    }

    mean_selected_q = (
        float(np.mean(selected_q_values))
        if selected_q_values
        else 0.0
    )

    mean_maximum_q = (
        float(np.mean(maximum_q_values))
        if maximum_q_values
        else 0.0
    )

    result = {
        "episode": episode_number,
        "seed": episode_seed,
        "mission": mission,
        "return": episode_return,
        "length": episode_length,
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "mean_selected_q_value": mean_selected_q,
        "mean_maximum_q_value": mean_maximum_q,
        "action_counts": readable_action_counts,
        "video": None,
    }

    return result, frames

def main():
    args = parse_args()
    validate_args(args)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(args.checkpoint)

    checkpoint, model = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    number_of_actions = model.number_of_actions

    navigation_actions_only = (
        number_of_actions == 3
    )

    if args.video_dir is not None:
        render_mode = "rgb_array"
        video_dir = Path(args.video_dir)
        video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    elif args.render:
        render_mode = "human"
        video_dir = None
    else:
        render_mode = None
        video_dir = None

    environment = make_env(
        render_mode=render_mode,
        rgb_partial_obs=True,
        navigation_actions_only=(
            navigation_actions_only
        ),
    )

    print("\n" + "=" * 72)
    print("DQN EVALUATION")
    print("=" * 72)
    print(f"Device:                    {device}")
    print(
        f"Algorithm:                 "
        f"{checkpoint['algorithm']}"
    )
    print(
        f"Checkpoint:                "
        f"{checkpoint_path.resolve()}"
    )
    print(
        "Checkpoint training steps: "
        f"{checkpoint['environment_steps']:,}"
    )
    print(f"Evaluation episodes:       {args.episodes}")
    print(f"First evaluation seed:     {args.seed}")
    print("Policy:                    greedy argmax Q")
    print(f"Number of actions:         {number_of_actions}")
    print("=" * 72)

    episode_results = []
    best_video_candidates = []

    try:
        for episode_index in range(args.episodes):
            episode_number = episode_index + 1
            episode_seed = args.seed + episode_index

            record_video = (
                video_dir is not None
                and args.video_episodes > 0
            )

            result, frames = evaluate_episode(
                model=model,
                environment=environment,
                episode_number=episode_number,
                episode_seed=episode_seed,
                device=device,
                record_video=record_video,
                show_steps=args.show_steps,
            )

            episode_results.append(result)

            if record_video:
                best_video_candidates.append(
                    {
                        "episode_index": episode_index,
                        "episode_number": episode_number,
                        "seed": episode_seed,
                        "return": result["return"],
                        "length": result["length"],
                        "frames": frames,
                    }
                )

                # Prefer higher return. If returns tie,
                # prefer a shorter episode.
                best_video_candidates.sort(
                    key=lambda candidate: (
                        -candidate["return"],
                        candidate["length"],
                        candidate["episode_number"],
                    )
                )

                del best_video_candidates[
                    args.video_episodes:
                ]

            print(
                f"episode={episode_number:3d} "
                f"seed={episode_seed:5d} "
                f"return={result['return']:7.3f} "
                f"length={result['length']:4d} "
                f"success={result['success']} "
                f"mean_Q={result['mean_selected_q_value']:8.4f} "
                f"mission={result['mission']!r}"
            )

    finally:
        environment.close()

    # Videos are encoded only after all episodes have
    # been evaluated and ranked.
    for rank, candidate in enumerate(
        best_video_candidates,
        start=1,
    ):
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

        result = episode_results[
            candidate["episode_index"]
        ]

        result["video"] = str(
            video_path.resolve()
        )
        result["video_rank"] = rank

    returns = np.asarray(
        [
            result["return"]
            for result in episode_results
        ],
        dtype=np.float64,
    )

    lengths = np.asarray(
        [
            result["length"]
            for result in episode_results
        ],
        dtype=np.float64,
    )

    successes = np.asarray(
        [
            float(result["success"])
            for result in episode_results
        ],
        dtype=np.float64,
    )

    selected_q_values = np.asarray(
        [
            result["mean_selected_q_value"]
            for result in episode_results
        ],
        dtype=np.float64,
    )

    total_action_counts = {}

    for result in episode_results:
        for action_name, count in (
            result["action_counts"].items()
        ):
            total_action_counts[action_name] = (
                total_action_counts.get(
                    action_name,
                    0,
                )
                + count
            )

    summary = {
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "algorithm": checkpoint["algorithm"],
        "checkpoint_environment_steps": checkpoint[
            "environment_steps"
        ],
        "checkpoint_episodes": checkpoint.get(
            "episodes"
        ),
        "evaluation_episodes": args.episodes,
        "first_evaluation_seed": args.seed,
        "last_evaluation_seed": (
            args.seed + args.episodes - 1
        ),
        "policy": "greedy_argmax_q",
        "number_of_actions": number_of_actions,
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(
            np.mean(lengths)
        ),
        "success_rate": float(
            np.mean(successes)
        ),
        "mean_selected_q_value": float(
            np.mean(selected_q_values)
        ),
        "total_action_counts": total_action_counts,
        "video_directory": (
            str(video_dir.resolve())
            if video_dir is not None
            else None
        ),
    }

    output = {
        "summary": summary,
        "episodes": episode_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("DQN EVALUATION SUMMARY")
    print("=" * 72)
    print(
        f"Mean return:               "
        f"{summary['mean_return']:.4f}"
    )
    print(
        f"Mean episode length:       "
        f"{summary['mean_episode_length']:.2f}"
    )
    print(
        f"Success rate:              "
        f"{summary['success_rate']:.2%}"
    )
    print(
        f"Mean selected Q-value:     "
        f"{summary['mean_selected_q_value']:.4f}"
    )
    print(
        f"Total action counts:       "
        f"{summary['total_action_counts']}"
    )
    print(
        f"Results saved to:          "
        f"{output_path.resolve()}"
    )

    if video_dir is not None:
        number_of_videos = min(
            args.episodes,
            args.video_episodes,
        )

        print(
            f"Saved best videos:         "
            f"{number_of_videos}"
        )
        print(
            f"Video directory:           "
            f"{video_dir.resolve()}"
        )

    print("=" * 72)

if __name__ == "__main__":
    main()
    