"""
Interactive Gradio demo: camera-based obstacle avoidance for a warehouse
robot (PPO + perception fusion + safety shield).

Run from the repository root:

    python app.py

then open http://127.0.0.1:7860 in a browser.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr

from src.app.sim_session import (
    CONTROLLER_CHOICES, PERCEPTION_CHOICES, SCENARIOS, SimSession,
)

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "metrics" / "navigation_eval.json"

_sessions: dict[str, SimSession] = {}


def get_sim(request: gr.Request) -> SimSession:
    key = getattr(request, "session_hash", None) or "default"
    if key not in _sessions:
        if len(_sessions) > 50:      # tiny guard against unbounded growth
            _sessions.pop(next(iter(_sessions)))
        _sessions[key] = SimSession()
    return _sessions[key]


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------
def _outputs(sim):
    return sim.frame(), sim.status_md(), sim.log_rows()


def on_load(request: gr.Request):
    return _outputs(get_sim(request))


def apply_settings(controller, perception, scenario, workers, forklifts,
                   pallets, boxes, reflective, request: gr.Request):
    sim = get_sim(request)
    sim.running = False
    sim.controller = controller
    sim.perception = perception
    sim.scenario = scenario
    sim.counts = {"workers": int(workers), "forklifts": int(forklifts),
                  "pallets": int(pallets), "boxes": int(boxes)}
    sim.reflective = float(reflective) / 100.0
    sim.reset()
    return _outputs(sim)


def new_layout(controller, perception, scenario, workers, forklifts,
               pallets, boxes, reflective, request: gr.Request):
    sim = get_sim(request)
    sim.running = False
    sim.controller, sim.perception, sim.scenario = (controller, perception,
                                                    scenario)
    sim.counts = {"workers": int(workers), "forklifts": int(forklifts),
                  "pallets": int(pallets), "boxes": int(boxes)}
    sim.reflective = float(reflective) / 100.0
    sim.reset(new_layout=True)
    return _outputs(sim)


def run(speed, auto_restart, request: gr.Request):
    sim = get_sim(request)
    sim.running = True
    if sim.done:
        sim.reset()
    fps = 12.0
    steps_per_frame = max(1, int(speed))
    while sim.running:
        t0 = time.time()
        for _ in range(steps_per_frame):
            sim.step()
            if sim.done:
                break
        yield _outputs(sim)
        if sim.done:
            if auto_restart:
                time.sleep(1.5)
                sim.reset(new_layout=True)
                yield _outputs(sim)
                continue
            sim.running = False
            yield _outputs(sim)
            break
        dt = time.time() - t0
        time.sleep(max(0.0, 1.0 / fps - dt))


def pause(request: gr.Request):
    sim = get_sim(request)
    sim.running = False
    return sim.status_md()


def single_step(request: gr.Request):
    sim = get_sim(request)
    sim.running = False
    if sim.done:
        sim.reset()
    sim.step()
    return _outputs(sim)


def reset_episode(request: gr.Request):
    sim = get_sim(request)
    sim.running = False
    sim.reset()
    return _outputs(sim)


def on_click(mode, evt: gr.SelectData, request: gr.Request):
    sim = get_sim(request)
    px, py = evt.index[0], evt.index[1]
    msg = sim.click(px, py, mode)
    frame, status, rows = _outputs(sim)
    return frame, status + "\n\n" + f"🖱️ {msg}", rows


def on_scenario(scenario):
    sc = SCENARIOS.get(scenario)
    if not sc:
        return [gr.update()] * 5
    return (gr.update(value=sc["workers"]), gr.update(value=sc["forklifts"]),
            gr.update(value=sc["pallets"]), gr.update(value=sc["boxes"]),
            gr.update(value=int(100 * sc.get("reflective", 0.4))))


# ----------------------------------------------------------------------
# Benchmark tab
# ----------------------------------------------------------------------
def run_benchmark(episodes, perception, progress=gr.Progress()):
    from src.evaluation.run_episodes import CONTROLLERS, evaluate
    rows = []
    names = ["ppo_shield", "ppo", "rule", "rule_shield"]
    for i, name in enumerate(names):
        progress((i + 0.5) / len(names), desc=CONTROLLERS[name])
        s, _ = evaluate(name, episodes=int(episodes), seed0=20_000,
                        perception=perception)
        rows.append([
            CONTROLLERS[name],
            f"{100 * s['success_rate']:.0f}%",
            f"{100 * s['collision_rate']:.0f}%",
            f"{100 * s['timeout_rate']:.0f}%",
            "-" if s["mean_time_to_goal_s"] is None
            else f"{s['mean_time_to_goal_s']:.1f}",
            f"{s['near_misses_per_episode']:.1f}",
        ])
    return rows


def saved_results_md():
    if not METRICS.exists():
        return ("_No saved evaluation yet. Run_ "
                "`python -m scripts.evaluate_navigation`.")
    rep = json.loads(METRICS.read_text())
    out = [f"**{rep.get('note', '')}** Episodes per row: "
           f"{rep.get('episodes')}.", "",
           "#### RL vs rule-based baseline (full perception)", "",
           "| Controller | Success | Collision | Timeout | "
           "Time-to-goal (s) |", "|---|---|---|---|---|"]
    names = {
        "ppo_shield": "PPO + safety shield (full system)",
        "ppo": "PPO only", "rule": "Rule-based baseline",
        "rule_shield": "Rule-based + shield"}
    for k, s in rep.get("controllers", {}).items():
        t = s["mean_time_to_goal_s"]
        out.append(f"| {names.get(k, k)} | {100*s['success_rate']:.0f}% | "
                   f"{100*s['collision_rate']:.0f}% | "
                   f"{100*s['timeout_rate']:.0f}% | "
                   f"{'-' if t is None else f'{t:.1f}'} |")
    out += ["", "#### Perception ablation (PPO + shield)", "",
            "| Perception | Success | Collision | Collisions by class |",
            "|---|---|---|---|"]
    for k, s in rep.get("ablation", {}).items():
        out.append(f"| {k} | {100*s['success_rate']:.0f}% | "
                   f"{100*s['collision_rate']:.0f}% | "
                   f"{s['collisions_by_class'] or '-'} |")
    if "reflective" in rep:
        out += ["", "#### Reflective hard-case scenes (PPO + shield)", "",
                "| Perception | Success | Collision |", "|---|---|---|"]
        for k, s in rep["reflective"].items():
            out.append(f"| {k} | {100*s['success_rate']:.0f}% | "
                       f"{100*s['collision_rate']:.0f}% |")
    if "shield_worker_step_out_stop_rate" in rep:
        out += ["", f"**Safety shield scripted test** (worker steps out at "
                f"~1 m): collision prevented in "
                f"{100*rep['shield_worker_step_out_stop_rate']:.0f}% "
                f"of scenes."]
    if "seeds" in rep:
        sd = rep["seeds"]
        out += ["", f"**Robustness over {len(sd['per_seed'])} training "
                f"seeds** (PPO + shield): success "
                f"{100*sd['success_mean']:.1f} ± {100*sd['success_std']:.1f}"
                f"%, collision {100*sd['collision_mean']:.1f} ± "
                f"{100*sd['collision_std']:.1f}%."]
    return "\n".join(out)


# ----------------------------------------------------------------------
# Real-image perception tab (optional heavy dependencies)
# ----------------------------------------------------------------------
_perception = {}


def analyse_image(image):
    if image is None:
        return None, "Upload an image first."
    try:
        from src.perception.live_pipeline import analyse
    except ImportError as exc:  # pragma: no cover
        return None, (f"Optional dependencies missing ({exc}). Install with "
                      "`pip install -r requirements-perception.txt`.")
    try:
        vis, md = analyse(image, _perception)
        return vis, md
    except Exception as exc:  # pragma: no cover
        return None, f"Perception failed: {exc}"


# ----------------------------------------------------------------------
INTRO = """
# 🤖 Camera-Based Obstacle Avoidance for Autonomous Warehouse Robots
**Group 3 · Image and Scene Recognition** — a PPO reinforcement-learning
policy drives a robot through a simulated warehouse using a camera-style
*obstacle map* (detector + monocular depth fusion, uncertainty, tracking).
A rule-based **safety shield** can override every action.
"""

HOW = """
### Pipeline (same design as the project document)

| Layer | In this demo |
|---|---|
| 1 · Sensing | Forward camera, 120° FOV, 6 m range (simulated) |
| 2 · Perception | **Detector branch**: labels *person, pallet, forklift* (can miss, recall 80–97 %). **Depth branch**: distance to *any* surface, with noise, false "shadow" returns and failures on shiny wrap. |
| 3 · Fusion & reasoning | Box-median depth for detected objects; depth-only returns become **unknown obstacles**; **uncertainty score** from temporal jumps + detector/depth disagreement + specular highlights; **tracking** (3-frame median + closing speed). |
| 4 · Decision & control | **PPO** (Stable-Baselines3, MLP 2×64) chooses GO / SLOW / STOP / TURN LEFT / TURN RIGHT. The **safety shield** checks: hard stop → person priority → uncertainty override, and logs every override. |

**State vector (47 values):** 21 fused depth rays, 7 sector distances,
7 sector classes (person / forklift), 7 closing speeds, path direction
(sin, cos), goal distance, speed, uncertainty.

**Reward:** + progress along the global path, +10 at the goal,
−10 collision, − near-miss (bigger margin for people), − harsh braking,
− small time cost.

**Global path + replanning:** the static rack map is known, so a grid
planner (Dijkstra distance field) gives the blue path. Pallets, boxes,
workers and forklifts are *not* on the map; when the camera sees them
blocking the path, the path is replanned around them (event-triggered).

**Domain randomisation:** every episode has a new rack layout, obstacle
counts & speeds, sensor noise, detector recall, shadow-return rate and
number of reflective pallets — the policy never sees raw pixels, only the
obstacle map, which is what makes sim-to-real transfer plausible.

**Colours on the map:** 🟫 racks · 🟧 workers · 🟨 forklifts ·
pallets (wood / light-blue = stretch-wrapped) · **?** boxes = unknown
objects the detector was never trained on · blue dashed = global path ·
rays green→red = fused distance.

*All navigation numbers come from simulation, not a physical robot.*
"""


def build_ui():
    with gr.Blocks(title="Warehouse Robot Obstacle Avoidance") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("🚚 Live simulation"):
            with gr.Row():
                with gr.Column(scale=3):
                    frame = gr.Image(label="Warehouse (click the map to "
                                     "place obstacles)", type="numpy",
                                     interactive=False, format="jpeg",
                                     show_label=True)
                    status = gr.Markdown()
                    with gr.Row():
                        run_btn = gr.Button("▶ Run", variant="primary")
                        pause_btn = gr.Button("⏸ Pause")
                        step_btn = gr.Button("⏭ Step")
                        reset_btn = gr.Button("↺ Restart episode")
                        layout_btn = gr.Button("🎲 New warehouse")
                    log = gr.Dataframe(
                        headers=["t (s)", "RL proposed", "Final action",
                                 "Shield reason"],
                        label="Safety-shield decision log (latest first)",
                        interactive=False, wrap=True)
                with gr.Column(scale=1, min_width=280):
                    controller = gr.Dropdown(
                        CONTROLLER_CHOICES, value="ppo_shield",
                        label="Controller")
                    perception = gr.Dropdown(
                        PERCEPTION_CHOICES, value="full",
                        label="Perception stack (ablation)")
                    scenario = gr.Dropdown(
                        list(SCENARIOS.keys()), value="Random warehouse",
                        label="Scenario")
                    click_mode = gr.Radio(
                        ["Unknown box (fallen carton)", "Worker (standing)",
                         "Shiny pallet (reflective)", "Remove obstacle"],
                        value="Unknown box (fallen carton)",
                        label="Click on the map places…")
                    speed = gr.Slider(1, 4, value=1, step=1,
                                      label="Simulation speed (× real time)")
                    auto = gr.Checkbox(False, label="Auto-restart with a new "
                                       "warehouse when an episode ends")
                    with gr.Accordion("Obstacle counts (domain "
                                      "randomisation)", open=False):
                        workers = gr.Slider(0, 8, value=3, step=1,
                                            label="Workers")
                        forklifts = gr.Slider(0, 3, value=1, step=1,
                                              label="Forklifts")
                        pallets = gr.Slider(0, 8, value=3, step=1,
                                            label="Pallets")
                        boxes = gr.Slider(0, 8, value=3, step=1,
                                          label="Unknown boxes")
                        reflective = gr.Slider(0, 100, value=40, step=10,
                                               label="Reflective pallets (%)")
                    apply_btn = gr.Button("Apply settings & restart")
            settings = [controller, perception, scenario, workers, forklifts,
                        pallets, boxes, reflective]
            outs = [frame, status, log]
            demo.load(on_load, None, outs)
            run_evt = run_btn.click(run, [speed, auto], outs)
            pause_btn.click(pause, None, status, cancels=[run_evt])
            step_btn.click(single_step, None, outs, cancels=[run_evt])
            reset_btn.click(reset_episode, None, outs, cancels=[run_evt])
            layout_btn.click(new_layout, settings, outs, cancels=[run_evt])
            apply_btn.click(apply_settings, settings, outs,
                            cancels=[run_evt])
            for comp in (controller, perception):
                comp.change(apply_settings, settings, outs,
                            cancels=[run_evt])
            scenario.change(on_scenario, scenario,
                            [workers, forklifts, pallets, boxes, reflective]
                            ).then(apply_settings, settings, outs,
                                   cancels=[run_evt])
            frame.select(on_click, [click_mode], outs)

        with gr.Tab("📊 Benchmark"):
            gr.Markdown("Run fresh evaluation episodes on unseen random "
                        "layouts and compare controllers.")
            with gr.Row():
                n_ep = gr.Slider(5, 100, value=20, step=5,
                                 label="Episodes per controller")
                b_perc = gr.Dropdown(PERCEPTION_CHOICES, value="full",
                                     label="Perception stack")
                b_btn = gr.Button("Run benchmark", variant="primary")
            table = gr.Dataframe(
                headers=["Controller", "Success", "Collision", "Timeout",
                         "Time-to-goal (s)", "Near-misses / ep"],
                interactive=False)
            b_btn.click(run_benchmark, [n_ep, b_perc], table)
            gr.Markdown("### Saved evaluation (100 held-out episodes)")
            gr.Markdown(saved_results_md())
            curve = ROOT / "results" / "graphs" / "ppo_training_curve.png"
            if curve.exists():
                gr.Image(str(curve), label="PPO training curve",
                         interactive=False)

        with gr.Tab("📷 Real image perception (optional)"):
            gr.Markdown(
                "Runs the real perception branch on an uploaded photo: "
                "**YOLO11n** detection + **Depth Anything V2-Small** "
                "relative depth + driving-corridor fusion + unknown-obstacle "
                "check. Needs `pip install -r requirements-perception.txt` "
                "and an internet connection the first time (model "
                "download). Uses `weights/yolo/best.pt` if present, "
                "otherwise COCO-pretrained `yolo11n.pt`.")
            with gr.Row():
                inp = gr.Image(type="numpy", label="Warehouse photo")
                outp = gr.Image(type="numpy", label="Perception result")
            p_md = gr.Markdown()
            gr.Button("Analyse", variant="primary").click(
                analyse_image, inp, [outp, p_md])

        with gr.Tab("ℹ️ How it works"):
            gr.Markdown(HOW)
    return demo


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="create a temporary public gradio.live link")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    demo = build_ui()
    demo.queue(default_concurrency_limit=8)
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
