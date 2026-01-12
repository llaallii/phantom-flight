"""
Training script for continuous-control RL algorithms (TD3, SAC, DDPG, PPO).

Usage examples:
    python -m src.train.train --algo sac --timesteps 500000
    python -m src.train.train --algo td3 --timesteps 1000000 --gui
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, ".")

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.hil_env import QuadHoverEnv

ALGO_MAP = {
    "ppo": PPO,
    "td3": TD3,
    "sac": SAC,
    "ddpg": DDPG,
}


def make_env(gui: bool = False, max_episode_steps: int = 1000) -> QuadHoverEnv:
    """Factory for creating the hover environment."""
    env = QuadHoverEnv(gui=gui, max_episode_steps=max_episode_steps)
    return Monitor(env)


def train(
    algo: str = "sac",
    total_timesteps: int = 500_000,
    gui: bool = False,
    seed: int = 42,
    log_dir: str = "outputs/logs",
    model_dir: str = "outputs/models",
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    hidden_sizes: tuple = (64, 64),
) -> None:
    """
    Train an RL agent to hover.

    Args:
        algo: One of 'ppo', 'td3', 'sac', 'ddpg'.
        total_timesteps: Total training steps.
        gui: If True, render PyBullet GUI during training (slow).
        seed: Random seed.
        log_dir: Directory for tensorboard logs.
        model_dir: Directory to save model checkpoints.
        eval_freq: Evaluate every N steps.
        checkpoint_freq: Save checkpoint every N steps.
        hidden_sizes: MLP hidden layer sizes (small for embedded).
    """
    algo_cls = ALGO_MAP.get(algo.lower())
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from {list(ALGO_MAP)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_{timestamp}"

    log_path = Path(log_dir) / run_name
    model_path = Path(model_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    # Vectorised training environment
    train_env = DummyVecEnv([lambda: make_env(gui=gui)])

    # Separate evaluation environment (headless)
    eval_env = DummyVecEnv([lambda: make_env(gui=False)])

    # Policy kwargs (small MLP for embedded deployment)
    policy_kwargs = dict(net_arch=list(hidden_sizes))

    # Algorithm-specific defaults
    common_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(log_path),
        policy_kwargs=policy_kwargs,
    )

    if algo.lower() in ("td3", "ddpg", "sac"):
        common_kwargs.update(
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
        )
    else:  # PPO
        common_kwargs.update(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )

    model = algo_cls(**common_kwargs)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_path),
        name_prefix=run_name,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_path),
        log_path=str(log_path),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Starting training: {algo.upper()} for {total_timesteps} steps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = model_path / f"{run_name}_final.zip"
    model.save(str(final_path))
    print(f"Training complete. Model saved to {final_path}")

    train_env.close()
    eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hover policy")
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=list(ALGO_MAP.keys()),
        help="RL algorithm",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000, help="Total training timesteps"
    )
    parser.add_argument("--gui", action="store_true", help="Render PyBullet GUI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        algo=args.algo,
        total_timesteps=args.timesteps,
        gui=args.gui,
        seed=args.seed,
    )
