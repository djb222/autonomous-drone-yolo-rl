import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_control.config import DroneConfig
from drone_control.env import DroneGoalEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    args = parser.parse_args()

    cfg = DroneConfig(max_episode_steps=3000)

    goals = np.array([
        [0.8, 0.0, 1.2],
        [0.8, 0.8, 1.3],
        [0.0, 0.8, 1.2],
        [0.0, 0.0, 1.1],
    ], dtype=float)

    env = DroneGoalEnv(goals=goals, gui=args.gui, cfg=cfg, render_sleep=False)
    env.reset()
    print("Starting PID waypoint demo. Goals:", goals.tolist())

    goal_index = 0

    try:
        for step in range(cfg.max_episode_steps):
            state = env.drone.get_state()
            goal = goals[goal_index]
            dist = float(np.linalg.norm(goal - state["pos"]))

            if dist < cfg.goal_radius:
                print(f"Reached goal {goal_index} at step {step}, dist={dist:.3f}")
                goal_index += 1
                if goal_index >= len(goals):
                    print("All goals reached.")
                    break
                goal = goals[goal_index]

            motor_forces = env.controller.compute_motor_forces(
                state=state,
                target_pos=goal,
                target_yaw=0.0,
                dt=cfg.dt,
            )

            env.drone.apply_motor_forces(motor_forces)
            p.stepSimulation(physicsClientId=env.client_id)

            if args.gui:
                time.sleep(cfg.dt)

            if step % 120 == 0:
                pos = state["pos"]
                euler = state["euler"]
                print(
                    f"step={step:04d} goal={goal_index} dist={dist:.3f} "
                    f"pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] "
                    f"rpy=[{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] "
                    f"motors={np.round(motor_forces, 2)}"
                )

            pos = env.drone.get_state()["pos"]
            if pos[2] < 0.05 or np.any(np.abs(pos) > cfg.world_limit):
                print("Safety stop: drone left the allowed area.")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
