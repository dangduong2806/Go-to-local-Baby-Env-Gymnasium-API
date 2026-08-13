"""Train the simplified VLN model using PPO.

Run a short debug experiment:

    python -m training.train_ppo --debug

Run a longer experiment:

    python -m training.train_ppo --num-updates 100
"""

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
import torch

from algorithms.gae import compute_gae
from algorithms.ppo import PPOTrainer
from algorithms.rollout import CollectorState, collect_rollout
from envs.make_env import make_env
from models.actor_critic import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/ppo_gotolocal",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def safe_mean(values):
    """Return zero when no episode completed during this rollout."""
    return statistics.mean(values) if values else 0.0


def save_checkpoint(
    path,
    model,
    optimizer,
    update,
    environment_steps,
    args,
):
    """Save the model and optimizer state for evaluation or resuming."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "number_of_actions": model.actor_head.out_features,
            "update": update,
            "environment_steps": environment_steps,
            "training_args": vars(args),
        },
        path,
    )


def save_training_history(history, path):
    """Save one row of metrics per completed PPO update."""
    if not history:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)


def print_debug_information(
    batch,
    advantages,
    returns,
):
    print("\nDebug tensor shapes")
    print("-------------------")
    print(f"images: {tuple(batch.images.shape)}")
    print(f"token_ids: {tuple(batch.token_ids.shape)}")
    print(
        "attention_masks: "
        f"{tuple(batch.attention_masks.shape)}"
    )
    print(f"directions: {tuple(batch.directions.shape)}")
    print(f"actions: {tuple(batch.actions.shape)}")
    print(f"rewards: {tuple(batch.rewards.shape)}")
    print(f"old_log_probs: {tuple(batch.old_log_probs.shape)}")
    print(f"values: {tuple(batch.values.shape)}")
    print(f"next_values: {tuple(batch.next_values.shape)}")
    print(f"advantages: {tuple(advantages.shape)}")
    print(f"returns: {tuple(returns.shape)}")

    rollout_tensors = vars(batch).values()

    assert not any(
        tensor.requires_grad for tensor in rollout_tensors
    )

    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()

    print("\nExample rollout entry")
    print("---------------------")
    print(f"image shape: {tuple(batch.images[0].shape)}")
    print(f"token IDs: {batch.token_ids[0]}")
    print(f"action: {batch.actions[0].item()}")
    print(f"reward: {batch.rewards[0].item():.4f}")
    print(f"terminated: {batch.terminated[0].item()}")
    print(f"truncated: {batch.truncated[0].item()}")
    print(
        f"old log probability: "
        f"{batch.old_log_probs[0].item():.4f}"
    )
    print(f"value: {batch.values[0].item():.4f}")
    print(f"next value: {batch.next_values[0].item():.4f}")

    print("\nFirst five GAE values")
    print("---------------------")
    print(f"advantages: {advantages[:5]}")
    print(f"returns: {returns[:5]}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = make_env(
        render_mode=None,
        rgb_partial_obs=True,
        seed=args.seed,
    )

    env.action_space.seed(args.seed)

    model = ActorCritic(
        number_of_actions=env.action_space.n,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=4,
        minibatch_size=64,
    )

    collector_state = CollectorState()
    environment_steps = 0
    history = []
    best_score = (-1.0, float("-inf"))

    try:
        for update in range(1, args.num_updates + 1):
            rollout, collector_state, episode_statistics = (
                collect_rollout(
                    env=env,
                    model=model,
                    tokenizer=model.tokenizer,
                    rollout_steps=args.rollout_steps,
                    device=device,
                    collector_state=collector_state,
                )
            )

            environment_steps += len(rollout)

            # GAE is calculated from detached rollout tensors.
            cpu_batch = rollout.as_tensors()

            advantages, returns = compute_gae(
                rewards=cpu_batch.rewards,
                values=cpu_batch.values,
                next_values=cpu_batch.next_values,
                terminated=cpu_batch.terminated,
                truncated=cpu_batch.truncated,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

            if args.debug and update == 1:
                assert len(rollout) == args.rollout_steps

                print_debug_information(
                    cpu_batch,
                    advantages,
                    returns,
                )

            metrics = trainer.update(
                rollout=rollout,
                advantages=advantages,
                returns=returns,
                debug=args.debug and update == 1,
            )

            mean_return = safe_mean(
                episode_statistics.episode_returns
            )
            mean_length = safe_mean(
                episode_statistics.episode_lengths
            )
            success_rate = safe_mean(
                episode_statistics.episode_successes
            )

            record = {
                "update": update,
                "environment_steps": environment_steps,
                "mean_episode_return": mean_return,
                "mean_episode_length": mean_length,
                "success_rate": success_rate,
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "entropy": metrics.entropy,
                "total_loss": metrics.total_loss,
                "approx_kl": metrics.approx_kl,
                "clip_fraction": metrics.clip_fraction,
                "explained_variance": metrics.explained_variance,
                "ratio_before_update": metrics.ratio_before_update,
            }
            history.append(record)

            save_training_history(
                history,
                output_dir / "training_metrics.csv",
            )

            # Save the most recent completed update so training progress is
            # available even if a later update is interrupted.
            save_checkpoint(
                path=output_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                update=update,
                environment_steps=environment_steps,
                args=args,
            )

            # Prefer higher success rate, then higher mean return when two
            # updates have the same success rate.
            current_score = (success_rate, mean_return)
            if current_score > best_score:
                best_score = current_score
                save_checkpoint(
                    path=output_dir / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    environment_steps=environment_steps,
                    args=args,
                )

            print(
                f"update={update:4d} "
                f"steps={environment_steps:7d} "
                f"return={mean_return:7.3f} "
                f"length={mean_length:6.1f} "
                f"success={success_rate:6.2%} "
                f"policy_loss={metrics.policy_loss:8.4f} "
                f"value_loss={metrics.value_loss:8.4f} "
                f"entropy={metrics.entropy:7.4f} "
                f"total_loss={metrics.total_loss:8.4f} "
                f"approx_kl={metrics.approx_kl:8.6f} "
                f"clip_fraction={metrics.clip_fraction:6.3f} "
                f"explained_variance="
                f"{metrics.explained_variance:7.3f}"
            )

        print(f"\nResults saved to: {output_dir.resolve()}")
        print(
            "Latest checkpoint: "
            f"{(output_dir / 'checkpoint_last.pt').resolve()}"
        )
        print(
            "Best checkpoint: "
            f"{(output_dir / 'checkpoint_best.pt').resolve()}"
        )
        print(
            "Training metrics: "
            f"{(output_dir / 'training_metrics.csv').resolve()}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
