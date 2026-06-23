# Autonomous Drone Goal Detection — YOLO + PyBullet + Reinforcement Learning

## Overview

This repository is a cleaned portfolio version of a university group project that combined drone simulation, reinforcement learning, and computer vision.

The system simulates a quadrotor in PyBullet, uses a Gymnasium-style environment for reinforcement learning, and integrates a YOLO object-detection pipeline to identify phone-like goal objects from a simulated drone camera.

The project is relevant to robotics, autonomous systems, simulation, computer vision, and software-based engineering workflows.

## My Contribution

This was a group project. My main contribution was the YOLO/perception side of the system and its integration into the simulation workflow.

I contributed to:

- Generating a synthetic YOLO dataset from the PyBullet simulation.
- Using PyBullet segmentation masks to automatically create YOLO-format bounding-box labels.
- Training a YOLO object detector for phone/goal-object detection.
- Integrating the trained YOLO model into the drone environment.
- Adding simulated camera capture from above the drone.
- Returning YOLO detection information through the environment `info` dictionary.
- Displaying annotated YOLO frames during live model testing.
- Testing detection confidence, object centering, camera field of view, and detection frequency.
- Debugging the perception/simulation interface.

## Technologies Used

- Python
- PyBullet
- Gymnasium
- Stable-Baselines3 PPO
- YOLO / Ultralytics
- OpenCV
- NumPy
- Reinforcement learning
- Computer vision
- Drone simulation

## Key Files

```text
├── generate_yolo_dataset.py      # Generates synthetic images and YOLO labels from PyBullet
├── yolo_test.py                  # Runs PPO model with YOLO detection enabled
├── yolo_phone_best.pt            # Trained YOLO model used for detection demo
├── 3GoalsV6DroneDirectionChangesV4.zip  # Trained PPO model used for demo
├── drone_control/
│   ├── env.py                    # Main environment with YOLO integration
│   ├── config.py                 # Drone/environment parameters
│   ├── pid.py                    # Cascaded PID controller
│   ├── mixer.py                  # Rotor force mixing
│   └── quadrotor.py              # PyBullet quadrotor wrapper
├── scripts/
│   ├── check_project.py          # Smoke test
│   ├── run_pid_demo.py           # PID waypoint demonstration
│   └── test_model.py             # PPO model evaluation without YOLO overlay
└── assets/                       # Drone URDF/STL assets
```

## YOLO Dataset Generation

The YOLO dataset generator uses PyBullet's rendered segmentation masks to automatically label the simulated phone/goal objects.

The process is:

```text
1. Reset drone position randomly.
2. Place a simulated top-down camera above the drone.
3. Render RGB and segmentation-mask images from PyBullet.
4. Use object IDs in the segmentation mask to locate phone/goal objects.
5. Convert the object mask into YOLO-format bounding-box labels.
6. Save images and labels into datasets/phone_dataset/.
```

Run dataset generation:

```bash
python generate_yolo_dataset.py
```

This creates a dataset structure similar to:

```text
datasets/phone_dataset/
├── images/train/
└── labels/train/
```

## YOLO Training

Example training command:

```bash
yolo detect train model=yolo11n.pt data=datasets/phone_dataset/data.yaml epochs=50 imgsz=640
```

The trained detector is saved by Ultralytics in:

```text
runs/detect/train/weights/best.pt
```

For the final demo, the trained model was renamed to:

```text
yolo_phone_best.pt
```

## Running the YOLO + PPO Demo

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the integrated YOLO demo:

```bash
python yolo_test.py
```

The script loads:

```text
3GoalsV6DroneDirectionChangesV4.zip
```

as the PPO navigation model, and:

```text
yolo_phone_best.pt
```

as the YOLO perception model.

During the run, the script prints detection information such as:

```text
YOLO detected phone conf=0.87 centered=True offset=(12.4, -8.2) goal_index=1 reward=...
```

It also displays an annotated OpenCV frame showing the detector output.

## How YOLO Was Integrated Into the Environment

YOLO is enabled through the `DroneGoalEnv` constructor:

```python
env = DroneGoalEnv(
    gui=True,
    enable_obstacles=True,
    render_sleep=False,
    enable_yolo=True,
    yolo_model_path="./yolo_phone_best.pt",
    yolo_conf_threshold=0.25,
    yolo_detection_interval=5,
    require_yolo_for_goal=False,
)
```

The environment captures camera frames from above the drone, runs YOLO at a reduced detection frequency, and stores the result in the environment information output.

The YOLO output includes:

- Whether an object was detected.
- The detected class name.
- Detection confidence.
- Bounding-box centre.
- Offset from image centre.
- Whether the object is approximately centred.
- An annotated frame for display.

## Engineering Relevance

This project demonstrates:

- Computer vision integration into a simulated robotic system.
- Synthetic dataset generation using simulation outputs.
- Practical use of YOLO for object detection.
- Python-based robotics software development.
- Integration between perception, simulation, and autonomous navigation.
- Debugging across multiple software layers.
- Communication of individual contribution within a group engineering project.

## Limitations and Future Work

The project was completed as a university prototype, so several improvements could be made:

- Improve realism of camera viewpoint and object appearance.
- Add lighting and texture randomisation to improve detector robustness.
- Use YOLO detections directly in the observation/state vector for retraining.
- Use detection confidence and image-centre offset more directly in the control policy.
- Evaluate detection performance using precision, recall, and confusion matrix metrics.
- Add a recorded demo video or GIF to make the project easier to assess quickly.

## Original Group Project

This is a cleaned portfolio version prepared to clearly show my contribution. The original university project was completed as a group project.
