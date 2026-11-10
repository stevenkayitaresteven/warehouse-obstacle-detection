# Camera-Based Obstacle Detection for Autonomous Warehouse Robots

## Project Overview

A computer-vision and robotics project developing a camera-based
perception and navigation system for autonomous warehouse robots.

The project combines object detection, monocular relative-depth
estimation, obstacle fusion, object tracking, reinforcement learning,
and safety constraints.

## System Architecture

1. Object detection: YOLO11n
2. Depth estimation: Depth Anything V2-Small
3. Obstacle fusion: YOLO detections associated with relative-depth maps
4. Unknown-obstacle candidates: Uncovered regions inside a driving corridor
5. Object tracking: Multi-frame tracking (planned)
6. Navigation: PPO with Stable-Baselines3 (planned)
7. Safety: Rule-based safety shield (planned)
8. Simulation: Gymnasium warehouse environment (planned)
9. Demonstration: Gradio interface (planned)

## Implemented Components

- GitHub-based project structure and development environment.
- Google Drive dataset storage and organization.
- LOCO dataset extraction and annotation processing.
- Conversion of LOCO COCO annotations into YOLO format.
- Validation of converted annotations.
- YOLO11n training on five LOCO warehouse object classes.
- Initial YOLO11n validation and diagnostic analysis.
- Depth Anything V2-Small inference.
- YOLO detection and relative-depth association.
- Driving-corridor mask generation.
- Candidate-region generation for further unknown-obstacle analysis.

## Current Dataset Classes

The current trained detector recognizes:

1. small_load_carrier
2. forklift
3. pallet
4. stillage
5. pallet_truck

Person detection is not yet integrated into this trained LOCO detector.

## Experimental Baseline

The initial YOLO11n training run completed with early stopping.

Best validation results:

| Metric | Result |
|---|---:|
| Precision | 41.4% |
| Recall | 35.0% |
| mAP@50 | 31.7% |
| mAP@50-95 | 13.69% |

These are baseline results, not final system performance.

The detector requires further improvement, particularly for forklifts
and pallet trucks.

Depth Anything V2 produces relative-depth estimates. These values
must not be interpreted as calibrated distances in metres.

## Repository Structure

- `notebooks/`: Reproducible experiments
- `scripts/`: Dataset preparation and training scripts
- `src/perception/`: Detection, depth estimation and obstacle fusion
- `src/simulation/`: Warehouse simulation
- `src/navigation/`: Reinforcement learning navigation
- `src/safety/`: Safety mechanisms
- `src/evaluation/`: Evaluation and metrics
- `src/app/`: Interactive demonstration
- `configs/`: Configuration files
- `tests/`: Software tests
- `datasets/`: Dataset directory placeholders
- `weights/`: Model weight directory placeholders
- `checkpoints/`: Training checkpoint placeholders
- `results/`: Experimental result directory placeholders

## Development Environment

- Python
- PyTorch
- Ultralytics YOLO
- Transformers
- OpenCV
- NumPy
- Gymnasium
- Stable-Baselines3
- Gradio
- Google Colab

## Data and Model Storage

Large datasets, pretrained weights, trained weights, checkpoints,
and generated experimental outputs are stored outside GitHub,
primarily in Google Drive.

## Academic Integrity

Measured results must be distinguished from targets and assumptions.
Software simulation results must be distinguished from physical
robot testing.
