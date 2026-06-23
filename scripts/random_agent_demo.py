import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_control.env import DroneGoalEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    args = parser.parse_args()

    env = DroneGoalEnv(gui=args.gui, render_sleep=args.gui)
    obs, info = env.reset(seed=7)

    total_reward = 0.0

    try:
        for step in range(1000):
            action = np.random.uniform(-1.0, 1.0, size=4)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 100 == 0:
                print(
                    f"step={step:04d} reward={reward:.3f} "
                    f"total={total_reward:.3f} dist={info['distance_to_goal']:.3f}"
                )

            if terminated or truncated:
                print("Episode finished.", info)
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
