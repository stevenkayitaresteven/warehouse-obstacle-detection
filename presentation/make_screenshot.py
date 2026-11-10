"""Render a demo frame (map + camera + HUD) for the title slide.

    python -m presentation.make_screenshot
"""
from pathlib import Path

from PIL import Image

from src.app.sim_session import SimSession

ROOT = Path(__file__).resolve().parents[1]


def main(layout_seed=21, steps=45):
    sim = SimSession()
    sim.scenario = "Fallen boxes (unknown objects) in the aisles"
    sim.layout_seed = sim.episode_seed = layout_seed
    sim.reset()
    for _ in range(steps):
        sim.step()
        if sim.done:
            break
    out = ROOT / "presentation" / "demo_screenshot.png"
    Image.fromarray(sim.frame()).save(out)
    print("saved", out, sim.outcome)


if __name__ == "__main__":
    main()
