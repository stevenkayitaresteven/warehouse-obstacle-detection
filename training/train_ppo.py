"""
Train the PPO navigation policy in the simulated warehouse.

Design (project Sections 8.3 and 10):
* State  : obstacle map from the simulated perception stack
* Actions: GO, SLOW, STOP, TURN_LEFT, TURN_RIGHT
* Reward : + progress / goal, - collision, - near-miss, - harsh braking
* Algo   : PPO (Stable-Baselines3), actor-critic MLP 2x64 (tanh)
* Sim-to-real: domain randomisation of layouts, obstacle counts, speeds,
  sensor noise, detector recall, reflective surfaces and shadow returns.

Usage
-----
    python -m training.train_ppo --timesteps 1500000 --seed 0

The trained model is written to ``models/ppo_warehouse_seed<k>.zip`` and the
actor is exported to ``models/ppo_warehouse_policy.npz`` (NumPy, used by
the demo so that the demo does not need PyTorch).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.navigation.controllers import export_sb3_to_numpy
from src.simulation.warehouse_env import make_env

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--init", default=None,
                    help="continue training from this SB3 .zip checkpoint")
    ap.add_argument("--export-npz", action="store_true",
                    help="also export this run's actor as the demo policy")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def wrapped(i):
        def _f():
            return Monitor(make_env(seed=args.seed * 1000 + i)())
        return _f

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = vec_cls([wrapped(i) for i in range(args.n_envs)])

    if args.init:
        model = PPO.load(args.init, env=venv, device="cpu")
        print(f"Continuing from {args.init} "
              f"({model.num_timesteps} steps done)")
    else:
        model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                           activation_fn=torch.nn.Tanh),
        seed=args.seed,
        verbose=1,
        device="cpu",
        tensorboard_log=None,
        )
    ckpt = CheckpointCallback(save_freq=max(200_000 // args.n_envs, 1),
                              save_path=str(ROOT / "checkpoints" / "ppo"),
                              name_prefix=f"ppo_seed{args.seed}")
    t0 = time.time()
    remaining = max(args.timesteps - model.num_timesteps, 0)
    model.learn(total_timesteps=remaining, callback=ckpt,
                progress_bar=False, reset_num_timesteps=args.init is None)
    elapsed = time.time() - t0

    zip_path = out / f"ppo_warehouse_seed{args.seed}.zip"
    model.save(zip_path)
    npz_path = out / f"ppo_warehouse_seed{args.seed}.npz"
    export_sb3_to_numpy(model, npz_path)
    if args.export_npz:
        export_sb3_to_numpy(model, out / "ppo_warehouse_policy.npz")

    meta = {
        "seed": args.seed,
        "timesteps": int(model.num_timesteps),
        "initialised_from": args.init,
        "n_envs": args.n_envs,
        "train_seconds": round(elapsed, 1),
        "algorithm": "PPO (Stable-Baselines3)",
        "policy": "MLP 2x64 tanh (actor & critic)",
    }
    (out / f"ppo_warehouse_seed{args.seed}.json").write_text(
        json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    venv.close()


if __name__ == "__main__":
    main()
