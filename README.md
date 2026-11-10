# Camera-Based Obstacle Detection for Autonomous Warehouse Robots

## Project Overview

A software-based autonomous warehouse robot perception and navigation project.

The system combines computer vision, depth estimation, obstacle mapping,
reinforcement learning, and safety constraints.

## Planned Architecture

1. Object Detection: YOLO11n
2. Depth Estimation: Depth Anything V2-Small
3. Obstacle Fusion: Known and unknown obstacle mapping
4. Object Tracking: Multi-frame tracking
5. Navigation: PPO using Stable-Baselines3
6. Safety: Rule-based safety shield
7. Simulation: Gymnasium-based warehouse environment
8. Demonstration: Gradio interface

## Repository Structure

- `notebooks/`: Experimental notebooks
- `src/perception/`: Computer vision components
- `src/simulation/`: Warehouse simulation
- `src/navigation/`: Reinforcement learning navigation
- `src/safety/`: Safety mechanisms
- `src/evaluation/`: Evaluation and metrics
- `src/app/`: Interactive demonstration
- `configs/`: Project configuration files
- `tests/`: Software tests
- `datasets/`: Dataset directory placeholders
- `weights/`: Model weight directory placeholders
- `checkpoints/`: Training checkpoint placeholders
- `results/`: Experimental results

## Development Environment

- Python
- PyTorch
- Ultralytics
- OpenCV
- Gymnasium
- Stable-Baselines3
- Gradio
- Google Colab

## Development Status

Environment setup completed.
Project structure created.
Implementation and experimental evaluation are pending.

## Academic Integrity

Performance targets must not be reported as measured results.
Simulation results must be distinguished from real-world testing.
