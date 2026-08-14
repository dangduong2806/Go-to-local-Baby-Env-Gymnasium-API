"""Plot PPO training and evaluation results.

Example:

    python -m scripts.plot_results \
        --training-csv runs/ppo_gotolocal_3actions/training_metrics.csv \
        --evaluation-json runs/ppo_gotolocal_3actions/evaluation_results.json \
        --output-dir runs/ppo_gotolocal_3actions/plots
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

# Use a non-interactive backend so plotting also works without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-csv",
        type=str,
        required=True,
        help="Path to training_metrics.csv.",
    )
    parser.add_argument(
        "--evaluation-json",
        type=str,
        required=True,
        help="Path to evaluation_results.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where PNG diagrams will be saved.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Window used for moving-average curves.",
    )

    return parser.parse_args()


def load_training_metrics(path):
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_evaluation_results(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_float(value):
    """Convert CSV text to float, returning NaN for missing values."""
    if value is None or value == "":
        return float("nan")

    try:
        return float(value)
    except ValueError:
        return float("nan")


def csv_series(rows, key):
    return np.asarray(
        [parse_float(row.get(key)) for row in rows],
        dtype=np.float64,
    )


def moving_average(values, window):
    """Calculate a trailing moving average while ignoring missing values."""
    values = np.asarray(values, dtype=np.float64)
    smoothed = np.full_like(values, np.nan)

    for index in range(len(values)):
        start = max(0, index - window + 1)
        selected_values = values[start : index + 1]
        finite_values = selected_values[
            np.isfinite(selected_values)
        ]

        if len(finite_values) > 0:
            smoothed[index] = np.mean(finite_values)

    return smoothed


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved: {path.resolve()}")


def plot_training_losses(rows, output_dir):
    environment_steps = csv_series(
        rows,
        "environment_steps",
    )
    policy_loss = csv_series(rows, "policy_loss")
    value_loss = csv_series(rows, "value_loss")
    total_loss = csv_series(rows, "total_loss")

    plt.figure(figsize=(10, 6))

    plt.plot(
        environment_steps,
        policy_loss,
        label="Policy loss",
        linewidth=1.8,
    )
    plt.plot(
        environment_steps,
        value_loss,
        label="Value loss",
        linewidth=1.8,
    )
    plt.plot(
        environment_steps,
        total_loss,
        label="Total loss",
        linewidth=1.8,
    )

    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.title("PPO Training Losses")
    plt.xlabel("Environment steps")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure(output_dir / "training_losses.png")


def plot_training_rewards(
    rows,
    output_dir,
    smooth_window,
):
    environment_steps = csv_series(
        rows,
        "environment_steps",
    )
    training_returns = csv_series(
        rows,
        "mean_episode_return",
    )
    validation_returns = csv_series(
        rows,
        "validation_mean_return",
    )

    smoothed_returns = moving_average(
        training_returns,
        smooth_window,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        environment_steps,
        training_returns,
        color="tab:blue",
        alpha=0.3,
        label="Training rollout return",
    )
    plt.plot(
        environment_steps,
        smoothed_returns,
        color="tab:blue",
        linewidth=2.2,
        label=f"Training moving average ({smooth_window})",
    )

    validation_mask = np.isfinite(validation_returns)

    if np.any(validation_mask):
        plt.plot(
            environment_steps[validation_mask],
            validation_returns[validation_mask],
            color="tab:orange",
            marker="o",
            linewidth=2,
            label="Validation return",
        )

    plt.title("Training and Validation Reward")
    plt.xlabel("Environment steps")
    plt.ylabel("Mean episode return")
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure(output_dir / "training_rewards.png")


def plot_training_success(
    rows,
    output_dir,
    smooth_window,
):
    environment_steps = csv_series(
        rows,
        "environment_steps",
    )

    training_success = (
        csv_series(rows, "success_rate") * 100.0
    )
    validation_success = (
        csv_series(rows, "validation_success_rate") * 100.0
    )

    smoothed_success = moving_average(
        training_success,
        smooth_window,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        environment_steps,
        training_success,
        color="tab:green",
        alpha=0.3,
        label="Training rollout success",
    )
    plt.plot(
        environment_steps,
        smoothed_success,
        color="tab:green",
        linewidth=2.2,
        label=f"Training moving average ({smooth_window})",
    )

    validation_mask = np.isfinite(validation_success)

    if np.any(validation_mask):
        plt.plot(
            environment_steps[validation_mask],
            validation_success[validation_mask],
            color="tab:red",
            marker="o",
            linewidth=2,
            label="Validation success",
        )

    plt.title("Training and Validation Success Rate")
    plt.xlabel("Environment steps")
    plt.ylabel("Success rate (%)")
    plt.ylim(-2, 102)
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure(output_dir / "training_success_rate.png")


def plot_evaluation_rewards(
    evaluation,
    output_dir,
    smooth_window,
):
    episodes = evaluation["episodes"]

    episode_numbers = np.asarray(
        [result["episode"] for result in episodes]
    )
    returns = np.asarray(
        [result["return"] for result in episodes],
        dtype=np.float64,
    )

    smoothed_returns = moving_average(
        returns,
        smooth_window,
    )

    plt.figure(figsize=(10, 6))

    plt.scatter(
        episode_numbers,
        returns,
        color="tab:blue",
        alpha=0.45,
        label="Episode return",
    )
    plt.plot(
        episode_numbers,
        smoothed_returns,
        color="tab:orange",
        linewidth=2.2,
        label=f"Moving average ({smooth_window})",
    )

    mean_return = float(np.mean(returns))

    plt.axhline(
        mean_return,
        color="tab:red",
        linestyle="--",
        label=f"Overall mean: {mean_return:.3f}",
    )

    plt.title("Evaluation Episode Rewards")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Episode return")
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure(output_dir / "evaluation_rewards.png")


def plot_evaluation_success(
    evaluation,
    output_dir,
    smooth_window,
):
    episodes = evaluation["episodes"]

    episode_numbers = np.asarray(
        [result["episode"] for result in episodes]
    )
    success_values = np.asarray(
        [
            100.0 if result["success"] else 0.0
            for result in episodes
        ],
        dtype=np.float64,
    )

    rolling_success = moving_average(
        success_values,
        smooth_window,
    )

    overall_success = float(np.mean(success_values))

    plt.figure(figsize=(10, 6))

    plt.scatter(
        episode_numbers,
        success_values,
        color="tab:green",
        alpha=0.3,
        label="Episode success",
    )
    plt.plot(
        episode_numbers,
        rolling_success,
        color="tab:green",
        linewidth=2.2,
        label=f"Rolling success rate ({smooth_window})",
    )
    plt.axhline(
        overall_success,
        color="tab:red",
        linestyle="--",
        label=f"Overall success: {overall_success:.1f}%",
    )

    plt.title("Evaluation Success Rate")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Success rate (%)")
    plt.ylim(-2, 102)
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure(output_dir / "evaluation_success_rate.png")


def main():
    args = parse_args()

    if args.smooth_window <= 0:
        raise ValueError(
            "--smooth-window must be greater than zero."
        )

    training_path = Path(args.training_csv)
    evaluation_path = Path(args.evaluation_json)
    output_dir = Path(args.output_dir)

    if not training_path.exists():
        raise FileNotFoundError(
            f"Training CSV does not exist: {training_path}"
        )

    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Evaluation JSON does not exist: {evaluation_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    training_rows = load_training_metrics(training_path)
    evaluation = load_evaluation_results(evaluation_path)

    if not training_rows:
        raise ValueError("Training CSV contains no rows.")

    if not evaluation.get("episodes"):
        raise ValueError(
            "Evaluation JSON contains no episode results."
        )

    plot_training_losses(
        rows=training_rows,
        output_dir=output_dir,
    )
    plot_training_rewards(
        rows=training_rows,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
    )
    plot_training_success(
        rows=training_rows,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
    )
    plot_evaluation_rewards(
        evaluation=evaluation,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
    )
    plot_evaluation_success(
        evaluation=evaluation,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
    )

    print(f"\nAll plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()