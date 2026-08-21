"""Train the VLN Q-network using Deep Q-Learning.

Short theory/debug run:

    python -m training.train_dqn --debug

Complete vanilla-DQN run:

    python -m training.train_dqn --total-steps 100000

Complete Double-DQN run:

    python -m training.train_dqn \
        --total-steps 100000 \
        --double-dqn
"""

import argparse
import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import matplotlib

# Use a non-interactive backend so plots work on headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from algorithms.dqn import (
    synchronize_target_network,
    update_dqn
)

from algorithms.replay_buffer import ReplayBuffer
from algorithms.rollout import observation_to_tensors
from envs.make_env import make_env
from models.q_network import QNetwork

@dataclass
class ProcessCounters:
    """Count the operations performed by the DQN system."""

    reporting_epoch: int = 0

    total_environment_samples: int = 0
    epoch_environment_samples: int = 0

    total_random_actions: int = 0
    epoch_random_actions: int = 0

    total_greedy_actions: int = 0
    epoch_greedy_actions: int = 0

    total_optimizer_updates: int = 0
    epoch_optimizer_updates: int = 0

    total_training_samples: int = 0
    epoch_training_samples: int = 0

    total_online_action_evaluations: int = 0
    epoch_online_action_evaluations: int = 0

    total_online_training_evaluations: int = 0
    epoch_online_training_evaluations: int = 0

    total_online_next_evaluations: int = 0
    epoch_online_next_evaluations: int = 0

    total_target_evaluations: int = 0
    epoch_target_evaluations: int = 0

    # The first synchronization happens before training.
    total_target_synchronizations: int = 1
    epoch_target_synchronizations: int = 0

    def reset_epoch(self):
        """Reset counters for the next reporting window."""
        self.epoch_environment_samples = 0
        self.epoch_random_actions = 0
        self.epoch_greedy_actions = 0
        self.epoch_optimizer_updates = 0
        self.epoch_training_samples = 0
        self.epoch_online_action_evaluations = 0

        self.epoch_online_training_evaluations = 0
        self.epoch_online_next_evaluations = 0
        self.epoch_target_evaluations = 0
        self.epoch_target_synchronizations = 0

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total-steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--train-frequency",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--target-update-frequency",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--validation-interval",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--validation-seed",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--plot-smooth-window",
        type=int,
        default=10,
        help="Number of reporting epochs in moving-average curves.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/dqn_gotolocal",
    )

    parser.add_argument(
        "--double-dqn",
        action="store_true",
        help="Use Double-DQN target calculation.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a short training pipeline test.",
    )

    return parser.parse_args()

def apply_debug_configuration(args):
    """Use a small configuration for a quick test."""

    if not args.debug:
        return

    args.total_steps = 1_000
    args.replay_capacity = 1_000
    args.learning_starts = 200
    args.batch_size = 32
    args.train_frequency = 4
    args.target_update_frequency = 100
    args.epsilon_decay_steps = 800
    args.validation_episodes = 5
    args.validation_interval = 500
    args.log_interval = 100


def validate_args(args):
    """Validate the training configuration."""

    positive_integer_values = {
        "total-steps": args.total_steps,
        "batch-size": args.batch_size,
        "replay-capacity": args.replay_capacity,
        "learning-starts": args.learning_starts,
        "train-frequency": args.train_frequency,
        "target-update-frequency": (
            args.target_update_frequency
        ),
        "epsilon-decay-steps": args.epsilon_decay_steps,
        "validation-episodes": args.validation_episodes,
        "validation-interval": args.validation_interval,
        "log-interval": args.log_interval,
        "plot-smooth-window": args.plot_smooth_window,

    }

    for name, value in positive_integer_values.items():
        if value <= 0:
            raise ValueError(
                f"--{name} must be greater than zero."
            )

    if args.learning_starts > args.total_steps:
        raise ValueError(
            "--learning-starts cannot exceed --total-steps."
        )

    if args.batch_size > args.replay_capacity:
        raise ValueError(
            "--batch-size cannot exceed --replay-capacity."
        )

    if args.learning_rate <= 0.0:
        raise ValueError(
            "--learning-rate must be greater than zero."
        )

    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(
            "--gamma must be between zero and one."
        )

    if args.max_gradient_norm <= 0.0:
        raise ValueError(
            "--max-gradient-norm must be greater than zero."
        )

    if not 0.0 <= args.epsilon_start <= 1.0:
        raise ValueError(
            "--epsilon-start must be between zero and one."
        )

    if not 0.0 <= args.epsilon_end <= 1.0:
        raise ValueError(
            "--epsilon-end must be between zero and one."
        )

    if args.epsilon_end > args.epsilon_start:
        raise ValueError(
            "--epsilon-end cannot exceed --epsilon-start."
        )

def safe_mean(values):
    """Return zero if the sequence is empty."""

    return statistics.mean(values) if values else 0.0

def calculate_epsilon(
    environment_step,
    epsilon_start,
    epsilon_end,
    epsilon_decay_steps, 
):
    """Calculate a linearly decaying epsilon."""
    progress = min(
        environment_step / epsilon_decay_steps,
        1.0,
    )

    return (
        epsilon_start
        + progress * (epsilon_end - epsilon_start)
    )

def select_training_action(
    model,
    observation,
    environment,
    epsilon,
    random_generator,
    device,   
):
    """Select an action using epsilon-greedy exploration."""
    if random_generator.random() < epsilon:
        action = environment.action_space.sample()
        return int(action), "random"

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

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            action = model.select_greedy_action(
                images=image,
                token_ids=token_ids,
                attention_mask=attention_mask,
                directions=direction,
            )
    finally:
        if was_training:
            model.train()

    return int(action.item()), "online-network"

def validate_policy(
    model,
    environment,
    number_of_episodes,
    base_seed,
    device,
):
    """Evaluate the greedy policy on fixed seeds."""
    was_training = model.training
    model.eval()

    episode_returns = []
    episode_lengths = []
    episode_successes = []

    try:
        with torch.no_grad():
            for episode in range(number_of_episodes):
                episode_seed = base_seed + episode

                observation, _ = environment.reset(
                    seed=episode_seed
                )

                terminated = False
                truncated = False
                episode_return = 0.0
                episode_length = 0

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

                    action = model.select_greedy_action(
                        images=image,
                        token_ids=token_ids,
                        attention_mask=attention_mask,
                        directions=direction,
                    )

                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        _,
                    ) = environment.step(
                        int(action.item())
                    )

                    episode_return += float(reward)
                    episode_length += 1

                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)

                episode_successes.append(
                    float(episode_return > 0.0)
                )

    finally:
        if was_training:
            model.train() 

    return {
        "mean_return": safe_mean(episode_returns),
        "mean_length": safe_mean(episode_lengths),
        "success_rate": safe_mean(episode_successes),
    }

def save_checkpoint(
    path,
    online_network,
    target_network,
    optimizer,
    environment_steps,
    episodes,
    epsilon,
    args,
):
    """Save the current DQN training state."""
    algorithm = (
        "double_dqn" if args.double_dqn else "dqn"
    )

    torch.save(
        {
            "algorithm": algorithm,
            "online_model_state_dict": (
                online_network.state_dict()
            ),
            "target_model_state_dict": (
                target_network.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "number_of_actions": (
                online_network.number_of_actions
            ),
            "environment_steps": environment_steps,
            "episodes": episodes,
            "epsilon": epsilon,
            "training_args": vars(args),
        },
        path,
    )

def save_training_history(history, path):
    """Save the training measurements to a CSV file."""

    if not history:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)


def history_series(history, key):
    """Convert one history column to a floating-point NumPy array."""

    return np.asarray(
        [
            float("nan")
            if record.get(key) is None
            else float(record[key])
            for record in history
        ],
        dtype=np.float64,
    )


def moving_average(values, window):
    """Calculate a trailing mean while ignoring missing values."""

    values = np.asarray(values, dtype=np.float64)
    smoothed = np.full_like(values, np.nan)

    for index in range(len(values)):
        start = max(0, index - window + 1)
        selected = values[start:index + 1]
        finite_values = selected[np.isfinite(selected)]

        if len(finite_values) > 0:
            smoothed[index] = np.mean(finite_values)

    return smoothed


def save_figure(figure, path):
    """Save and close one Matplotlib figure."""

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_training_history(
    history,
    output_dir,
    smooth_window,
):
    """Draw DQN losses, rewards, success, and Q-value curves."""

    if not history:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    environment_steps = history_series(
        history,
        "environment_steps",
    )
    optimizer_updates = history_series(
        history,
        "optimizer_updates",
    )

    # TD loss and absolute TD error are training measurements. Validation
    # evaluates task performance and therefore has no replay-based TD loss.
    td_loss = history_series(history, "mean_td_loss")
    absolute_td_error = history_series(
        history,
        "mean_absolute_td_error",
    )
    learning_mask = (
        np.isfinite(td_loss)
        & np.isfinite(optimizer_updates)
        & (optimizer_updates > 0)
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    if np.any(learning_mask):
        axis.plot(
            environment_steps[learning_mask],
            td_loss[learning_mask],
            color="tab:blue",
            linewidth=2.0,
            label="Mean Huber TD loss",
        )
        axis.plot(
            environment_steps[learning_mask],
            absolute_td_error[learning_mask],
            color="tab:orange",
            linewidth=1.8,
            label="Mean absolute TD error",
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No optimizer updates recorded yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_title("DQN Training Loss and TD Error")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.3)

    if np.any(learning_mask):
        axis.legend()

    save_figure(
        figure,
        plots_dir / "training_td_loss.png",
    )

    training_returns = history_series(
        history,
        "mean_episode_return",
    )
    validation_returns = history_series(
        history,
        "validation_mean_return",
    )
    smoothed_returns = moving_average(
        training_returns,
        smooth_window,
    )
    validation_mask = np.isfinite(validation_returns)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        environment_steps,
        training_returns,
        color="tab:blue",
        alpha=0.3,
        label="Training episode return",
    )
    axis.plot(
        environment_steps,
        smoothed_returns,
        color="tab:blue",
        linewidth=2.2,
        label=(
            f"Training moving average "
            f"({smooth_window} reporting epochs)"
        ),
    )

    if np.any(validation_mask):
        axis.plot(
            environment_steps[validation_mask],
            validation_returns[validation_mask],
            color="tab:orange",
            marker="o",
            linewidth=2.0,
            label="Fixed-seed validation return",
        )

    axis.set_title("DQN Training and Validation Reward")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Mean episode return")
    axis.grid(alpha=0.3)
    axis.legend()
    save_figure(
        figure,
        plots_dir / "training_rewards.png",
    )

    training_success = (
        history_series(history, "success_rate") * 100.0
    )
    validation_success = (
        history_series(
            history,
            "validation_success_rate",
        )
        * 100.0
    )
    smoothed_success = moving_average(
        training_success,
        smooth_window,
    )
    validation_success_mask = np.isfinite(
        validation_success
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        environment_steps,
        training_success,
        color="tab:green",
        alpha=0.3,
        label="Training success rate",
    )
    axis.plot(
        environment_steps,
        smoothed_success,
        color="tab:green",
        linewidth=2.2,
        label=(
            f"Training moving average "
            f"({smooth_window} reporting epochs)"
        ),
    )

    if np.any(validation_success_mask):
        axis.plot(
            environment_steps[validation_success_mask],
            validation_success[validation_success_mask],
            color="tab:red",
            marker="o",
            linewidth=2.0,
            label="Fixed-seed validation success",
        )

    axis.set_title("DQN Training and Validation Success Rate")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Success rate (%)")
    axis.set_ylim(-2, 102)
    axis.grid(alpha=0.3)
    axis.legend()
    save_figure(
        figure,
        plots_dir / "training_success_rate.png",
    )

    mean_q_values = history_series(
        history,
        "mean_q_value",
    )
    mean_target_values = history_series(
        history,
        "mean_target_value",
    )
    q_mask = (
        np.isfinite(mean_q_values)
        & np.isfinite(mean_target_values)
        & np.isfinite(optimizer_updates)
        & (optimizer_updates > 0)
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    if np.any(q_mask):
        axis.plot(
            environment_steps[q_mask],
            mean_q_values[q_mask],
            color="tab:purple",
            linewidth=2.0,
            label="Mean selected online Q(s,a)",
        )
        axis.plot(
            environment_steps[q_mask],
            mean_target_values[q_mask],
            color="tab:brown",
            linewidth=2.0,
            label="Mean Bellman target",
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No Q-learning updates recorded yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.5,
    )
    axis.set_title("DQN Online Q-Values and Bellman Targets")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.3)

    if np.any(q_mask):
        axis.legend()

    save_figure(
        figure,
        plots_dir / "training_q_values.png",
    )

    print(f"[PLOTS] Updated training plots in: {plots_dir.resolve()}")

def print_initialization(
    args,
    device,
    number_of_actions, 
):
    """Explain the initial DQN configuration."""

    algorithm = (
        "Double DQN" if args.double_dqn else "Vanilla DQN"
    )

    print("\n" + "=" * 72)
    print("DQN TRAINING INITIALIZATION")
    print("=" * 72)
    print(f"Device:                       {device}")
    print(f"Algorithm:                    {algorithm}")
    print(f"Number of actions:            {number_of_actions}")
    print(f"Total environment steps:      {args.total_steps:,}")
    print(f"Replay-buffer capacity:       {args.replay_capacity:,}")
    print(f"Replay warm-up samples:       {args.learning_starts:,}")
    print(f"Training samples per update:  {args.batch_size:,}")
    print(
        "Training frequency:           "
        f"every {args.train_frequency} environment steps"
    )
    print(
        "Target synchronization:       "
        f"every {args.target_update_frequency:,} "
        "environment steps"
    )
    print(f"Discount factor gamma:        {args.gamma}")
    print(f"Learning rate:                {args.learning_rate}")
    print(f"Starting epsilon:             {args.epsilon_start}")
    print(f"Final epsilon:                {args.epsilon_end}")
    print(
        "Epsilon decay duration:       "
        f"{args.epsilon_decay_steps:,} steps"
    )

    print("\nNetwork responsibilities")
    print("------------------------")
    print(
        "Online network: predicts Q-values and receives "
        "gradient updates."
    )
    print(
        "Target network: calculates stable next-state "
        "Q-learning targets."
    )
    print(
        "Replay buffer: stores transitions and returns "
        "random minibatches."
    )

    print("\nInitial synchronization")
    print("-----------------------")
    print("target network <- online network")

    print("\nTerminology")
    print("-----------")
    print(
        "A reporting epoch is a screen-reporting window, "
        "not one pass through a fixed dataset."
    )
    print(
        "DQN samples randomly from replay memory, so the "
        "same transition may be reused."
    )
    print("=" * 72)

def print_training_report(
    counters,
    environment_step,
    args,
    replay_buffer,
    number_of_actions,
    completed_episodes,
    epsilon,
    mean_return,
    mean_length,
    success_rate,
    mean_loss,
    mean_q_value,
    mean_target_value,
    mean_td_error,
    mean_gradient_norm,
    validation_metrics,
):
    """Print one detailed DQN process report."""

    counters.reporting_epoch += 1

    first_step = (
        environment_step
        - counters.epoch_environment_samples
        + 1
    )

    phase = (
        "REPLAY WARM-UP"
        if environment_step < args.learning_starts
        else "Q-LEARNING"
    )

    current_q_outputs = (
        counters.epoch_online_training_evaluations
        * number_of_actions
    )

    target_q_outputs = (
        counters.epoch_target_evaluations
        * number_of_actions
    )

    print("\n" + "=" * 72)
    print(
        f"DQN REPORTING EPOCH {counters.reporting_epoch}"
    )
    print("=" * 72)
    print(f"Current phase:                {phase}")
    print(
        "Reporting step range:         "
        f"{first_step:,}-{environment_step:,}"
    )
    print(
        "Total training progress:      "
        f"{environment_step:,}/{args.total_steps:,}"
    )
    print(
        "Environment samples epoch:    "
        f"{counters.epoch_environment_samples:,}"
    )
    print(
        "Environment samples total:    "
        f"{counters.total_environment_samples:,}"
    )
    print(f"Completed episodes:           {completed_episodes:,}")

    print("\n1. Epsilon-greedy experience collection")
    print("---------------------------------------")
    print(f"Current epsilon:              {epsilon:.4f}")
    print(
        "Random actions this epoch:    "
        f"{counters.epoch_random_actions:,}"
    )
    print(
        "Online-network actions:       "
        f"{counters.epoch_greedy_actions:,}"
    )
    print(
        "Replay-buffer occupancy:      "
        f"{len(replay_buffer):,}/{args.replay_capacity:,}"
    )

    print("\n2. Online network")
    print("-----------------")
    print(
        "States used to choose actions:"
        f" {counters.epoch_online_action_evaluations:,}"
    )
    print(
        "Training states evaluated:    "
        f"{counters.epoch_online_training_evaluations:,}"
    )
    print(
        "Current-state Q-values made:  "
        f"{current_q_outputs:,}"
    )

    if args.double_dqn:
        print(
            "Next states used to choose "
            "actions:                     "
            f"{counters.epoch_online_next_evaluations:,}"
        )

    print(
        "Optimizer updates epoch:      "
        f"{counters.epoch_optimizer_updates:,}"
    )
    print(
        "Optimizer updates total:      "
        f"{counters.total_optimizer_updates:,}"
    )

    print("\n3. Replay-buffer training samples")
    print("---------------------------------")
    print(
        "Samples in one minibatch:     "
        f"{args.batch_size:,}"
    )
    print(
        "Samples presented this epoch: "
        f"{counters.epoch_training_samples:,}"
    )
    print(
        "Samples presented total:      "
        f"{counters.total_training_samples:,}"
    )
    print(
        "These count presentations, not necessarily "
        "unique transitions."
    )

    print("\n4. Target network")
    print("-----------------")
    print(
        "Next states evaluated:        "
        f"{counters.epoch_target_evaluations:,}"
    )
    print(
        "Next-state Q-values made:     "
        f"{target_q_outputs:,}"
    )
    print(
        "Q-learning targets calculated:"
        f" {counters.epoch_target_evaluations:,}"
    )
    print(
        "Synchronizations this epoch:  "
        f"{counters.epoch_target_synchronizations:,}"
    )
    print(
        "Synchronizations total:       "
        f"{counters.total_target_synchronizations:,}"
    )

    print("\n5. Learning measurements")
    print("------------------------")
    print(f"Mean episode return:          {mean_return:.6f}")
    print(f"Mean episode length:          {mean_length:.2f}")
    print(f"Success rate:                 {success_rate:.2%}")
    print(f"Mean Huber TD loss:           {mean_loss:.6f}")
    print(f"Mean selected Q(s,a):         {mean_q_value:.6f}")
    print(f"Mean target value:            {mean_target_value:.6f}")
    print(f"Mean absolute TD error:       {mean_td_error:.6f}")
    print(f"Mean gradient norm:           {mean_gradient_norm:.6f}")

    print("\n6. Elementary DQN theory")
    print("------------------------")
    print("Online prediction:")
    print("  Q_online(state, action_taken)")
    print("Q-learning target:")
    print(
        "  reward + gamma * (1 - terminated) "
        "* max Q_target(next_state, action)"
    )
    print("Temporal-difference error:")
    print("  target - online prediction")
    print("Optimization:")
    print("  minimize Huber loss")
    print("Network updated:")
    print("  online network only")

    if args.double_dqn:
        print("\nDouble-DQN target:")
        print(
            "  online network selects the next action"
        )
        print(
            "  target network evaluates the selected action"
        )

    if environment_step < args.learning_starts:
        samples_remaining = (
            args.learning_starts - environment_step
        )

        print("\nWarm-up explanation")
        print("-------------------")
        print(
            "No optimizer updates are expected during "
            "replay warm-up."
        )
        print(
            "Samples remaining before learning: "
            f"{samples_remaining:,}"
        )

    if validation_metrics is not None:
        print("\n7. Fixed-seed validation")
        print("------------------------")
        print(
            "Validation mean return:       "
            f"{validation_metrics['mean_return']:.6f}"
        )
        print(
            "Validation mean length:       "
            f"{validation_metrics['mean_length']:.2f}"
        )
        print(
            "Validation success rate:      "
            f"{validation_metrics['success_rate']:.2%}"
        )

    print("=" * 72)

def main():
    args = parse_args()
    apply_debug_configuration(args)
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    random_generator = random.Random(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    environment = make_env(
        render_mode=None,
        rgb_partial_obs=True,
        navigation_actions_only=True,
        seed=args.seed,
    )

    validation_environment = make_env(
        render_mode=None,
        rgb_partial_obs=True,
        navigation_actions_only=True,
        seed=args.validation_seed,
    )

    environment.action_space.seed(args.seed)

    number_of_actions = environment.action_space.n

    online_network = QNetwork(
        number_of_actions=number_of_actions,
    ).to(device)

    target_network = QNetwork(
        number_of_actions=number_of_actions,
    ).to(device)

    synchronize_target_network(
        online_network=online_network,
        target_network=target_network
    )

    online_network.train()

    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=args.learning_rate,
    )

    replay_buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        seed=args.seed
    )

    counters = ProcessCounters()

    print_initialization(
        args=args,
        device=device,
        number_of_actions=number_of_actions
    )

    observation, _ = environment.reset(seed=args.seed)

    episode_return = 0.0
    episode_length = 0
    completed_episodes = 0

    recent_returns = []
    recent_lengths = []
    recent_successes = []
    recent_updates = []

    history = []

    best_score = (
        -1.0,
        float("-inf"),
        float("-inf"),
    )

    try:
        for environment_step in range(
            1,
            args.total_steps + 1,
        ):
            epsilon = calculate_epsilon(
                environment_step=environment_step,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=(
                    args.epsilon_decay_steps
                ),
            )

            action, action_source = select_training_action(
                model=online_network,
                observation=observation,
                environment=environment,
                epsilon=epsilon,
                random_generator=random_generator,
                device=device
            )

            if action_source == "random":
                counters.total_random_actions += 1
                counters.epoch_random_actions += 1
            else:
                counters.total_greedy_actions += 1
                counters.epoch_greedy_actions += 1

                counters.total_online_action_evaluations += 1
                counters.epoch_online_action_evaluations += 1

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = environment.step(action)

            replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated
            )

            counters.total_environment_samples += 1
            counters.epoch_environment_samples += 1

            episode_return += float(reward)
            episode_length += 1

            if terminated or truncated:
                completed_episodes += 1

                recent_returns.append(episode_return)
                recent_lengths.append(episode_length)
                recent_successes.append(
                    float(episode_return > 0.0)
                )

                observation, _ = environment.reset()

                episode_return = 0.0
                episode_length = 0
            else:
                observation = next_observation

            ready_to_train = (
                environment_step >= args.learning_starts
                and len(replay_buffer) >= args.batch_size
            )

            should_update = (
                ready_to_train 
                and environment_step % args.train_frequency == 0
            )

            if should_update:
                batch = replay_buffer.sample(
                    batch_size=args.batch_size,
                    tokenizer=online_network.tokenizer,
                    device=device,
                )

                update_result = update_dqn(
                    online_network=online_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    batch=batch,
                    gamma=args.gamma,
                    max_gradient_norm=(
                        args.max_gradient_norm
                    ),
                    double_dqn=args.double_dqn,
                )

                recent_updates.append(update_result)

                counters.total_optimizer_updates += 1
                counters.epoch_optimizer_updates += 1

                counters.total_training_samples += (
                    args.batch_size
                )

                counters.epoch_training_samples += (
                    args.batch_size
                )

                counters.total_online_training_evaluations += (
                    args.batch_size
                )
                counters.epoch_online_training_evaluations += (
                    args.batch_size
                )

                counters.total_target_evaluations += (
                    args.batch_size
                )
                counters.epoch_target_evaluations += (
                    args.batch_size
                )

                if args.double_dqn:
                    counters.total_online_next_evaluations += (
                        args.batch_size
                    )
                    counters.epoch_online_next_evaluations += (
                        args.batch_size
                    )
            
            should_synchronize = (
                ready_to_train
                and environment_step % args.target_update_frequency == 0
            )

            if should_synchronize:
                synchronize_target_network(
                    online_network=online_network,
                    target_network=target_network,
                )

                counters.total_target_synchronizations += 1
                counters.epoch_target_synchronizations += 1

                print(
                    "\n[TARGET SYNCHRONIZATION] "
                    f"step={environment_step:,}: "
                    "target network <- online network"
                )

            should_validate = (
                environment_step
                % args.validation_interval
                == 0
                or environment_step == args.total_steps
            )

            validation_metrics = None

            if should_validate:
                print(
                    "\n[VALIDATION] Evaluating the greedy "
                    "online policy on fixed seeds..."
                )

                validation_metrics = validate_policy(
                    model=online_network,
                    environment=validation_environment,
                    number_of_episodes=(
                        args.validation_episodes
                    ),
                    base_seed=args.validation_seed,
                    device=device,
                )


                current_score = (
                    validation_metrics["success_rate"],
                    validation_metrics["mean_return"],
                    -validation_metrics["mean_length"],
                )

                if current_score > best_score:
                    best_score = current_score

                    save_checkpoint(
                        path=(
                            output_dir
                            / "checkpoint_best.pt"
                        ),
                        online_network=online_network,
                        target_network=target_network,
                        optimizer=optimizer,
                        environment_steps=environment_step,
                        episodes=completed_episodes,
                        epsilon=epsilon,
                        args=args,
                    )

                    print(
                        "[CHECKPOINT] New best checkpoint "
                        f"saved at step {environment_step:,}: "
                        "validation success="
                        f"{validation_metrics['success_rate']:.2%}"
                    )

            should_report = (
                environment_step % args.log_interval == 0
                or should_validate
                or environment_step == args.total_steps
            )

            if should_report:
                mean_return = safe_mean(recent_returns)
                mean_length = safe_mean(recent_lengths)
                success_rate = safe_mean(recent_successes)

                mean_loss = safe_mean(
                    [
                        result.loss
                        for result in recent_updates
                    ]
                )

                mean_q_value = safe_mean(
                    [
                        result.mean_q_value
                        for result in recent_updates
                    ]
                )

                mean_target_value = safe_mean(
                    [
                        result.mean_target_value
                        for result in recent_updates
                    ]
                )

                mean_td_error = safe_mean(
                    [
                        result.mean_absolute_td_error
                        for result in recent_updates
                    ]
                )

                mean_gradient_norm = safe_mean(
                    [
                        result.gradient_norm
                        for result in recent_updates
                    ]
                )

                record = {
                    "reporting_epoch": (
                        counters.reporting_epoch + 1
                    ),
                    "environment_steps": environment_step,
                    "episodes": completed_episodes,
                    "buffer_size": len(replay_buffer),
                    "epsilon": epsilon,
                    "random_actions": (
                        counters.epoch_random_actions
                    ),
                    "greedy_actions": (
                        counters.epoch_greedy_actions
                    ),
                    "optimizer_updates": (
                        counters.epoch_optimizer_updates
                    ),
                    "training_samples_presented": (
                        counters.epoch_training_samples
                    ),
                    "target_synchronizations": (
                        counters.epoch_target_synchronizations
                    ),
                    "mean_episode_return": mean_return,
                    "mean_episode_length": mean_length,
                    "success_rate": success_rate,
                    "mean_td_loss": mean_loss,
                    "mean_q_value": mean_q_value,
                    "mean_target_value": mean_target_value,
                    "mean_absolute_td_error": mean_td_error,
                    "mean_gradient_norm": mean_gradient_norm,
                    "validation_mean_return": (
                        validation_metrics["mean_return"]
                        if validation_metrics is not None
                        else None
                    ),
                    "validation_mean_length": (
                        validation_metrics["mean_length"]
                        if validation_metrics is not None
                        else None
                    ),
                    "validation_success_rate": (
                        validation_metrics["success_rate"]
                        if validation_metrics is not None
                        else None
                    ),
                }

                history.append(record)

                save_training_history(
                    history=history,
                    path=(
                        output_dir
                        / "training_metrics.csv"
                    ),
                )

                if should_validate:
                    plot_training_history(
                        history=history,
                        output_dir=output_dir,
                        smooth_window=(
                            args.plot_smooth_window
                        ),
                    )

                print_training_report(
                    counters=counters,
                    environment_step=environment_step,
                    args=args,
                    replay_buffer=replay_buffer,
                    number_of_actions=number_of_actions,
                    completed_episodes=completed_episodes,
                    epsilon=epsilon,
                    mean_return=mean_return,
                    mean_length=mean_length,
                    success_rate=success_rate,
                    mean_loss=mean_loss,
                    mean_q_value=mean_q_value,
                    mean_target_value=mean_target_value,
                    mean_td_error=mean_td_error,
                    mean_gradient_norm=mean_gradient_norm,
                    validation_metrics=validation_metrics,
                )

                recent_returns.clear()
                recent_lengths.clear()
                recent_successes.clear()
                recent_updates.clear()

                counters.reset_epoch()

            if should_validate:
                save_checkpoint(
                    path=output_dir / "checkpoint_last.pt",
                    online_network=online_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    environment_steps=environment_step,
                    episodes=completed_episodes,
                    epsilon=epsilon,
                    args=args,
                )

        print("\n" + "=" * 72)
        print("DQN TRAINING COMPLETED")
        print("=" * 72)
        print(
            "Environment samples:         "
            f"{counters.total_environment_samples:,}"
        )
        print(
            "Completed episodes:          "
            f"{completed_episodes:,}"
        )
        print(
            "Optimizer updates:           "
            f"{counters.total_optimizer_updates:,}"
        )
        print(
            "Training samples presented:  "
            f"{counters.total_training_samples:,}"
        )
        print(
            "Target synchronizations:     "
            f"{counters.total_target_synchronizations:,}"
        )
        print(f"Results directory:           {output_dir.resolve()}")
        print(
            "Latest checkpoint:           "
            f"{(output_dir / 'checkpoint_last.pt').resolve()}"
        )
        print(
            "Best checkpoint:             "
            f"{(output_dir / 'checkpoint_best.pt').resolve()}"
        )
        print(
            "Training metrics:            "
            f"{(output_dir / 'training_metrics.csv').resolve()}"
        )
        print(
            "Training plots:              "
            f"{(output_dir / 'plots').resolve()}"
        )
        print("=" * 72)

    finally:
        environment.close()
        validation_environment.close()


if __name__ == "__main__":
    main()
