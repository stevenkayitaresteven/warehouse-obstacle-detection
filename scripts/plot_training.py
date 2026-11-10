"""
Plot the PPO training curve from the Stable-Baselines3 console log.

    python -m scripts.plot_training logs/train_seed0.log
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(log_path):
    rows, cur = [], {}
    for line in Path(log_path).read_text().splitlines():
        m = re.match(r"\|\s+(\w+)\s+\|\s+([-\d.e+]+)\s+\|", line)
        if m:
            cur[m.group(1)] = float(m.group(2))
            if m.group(1) == "total_timesteps":
                if "ep_rew_mean" in cur:
                    rows.append(dict(cur))
                cur = {}
    return rows


def main():
    logs = sys.argv[1:] or [str(ROOT / "logs" / "train_seed0.log")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    out_dir = ROOT / "results" / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for lp in logs:
        rows = parse(lp)
        if not rows:
            continue
        name = Path(lp).stem.replace("train_", "")
        t = [r["total_timesteps"] / 1e6 for r in rows]
        ax[0].plot(t, [r["ep_rew_mean"] for r in rows], label=name)
        ax[1].plot(t, [r["ep_len_mean"] * 0.2 for r in rows], label=name)
        with open(ROOT / "results" / "metrics" / f"training_{name}.csv",
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["total_timesteps",
                                              "ep_rew_mean", "ep_len_mean"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in w.fieldnames})
    ax[0].set_title("PPO: mean episode return")
    ax[1].set_title("PPO: mean episode duration (s)")
    for a in ax:
        a.set_xlabel("environment steps (millions)")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    p = out_dir / "ppo_training_curve.png"
    fig.savefig(p, dpi=130)
    print("saved", p)


if __name__ == "__main__":
    main()
