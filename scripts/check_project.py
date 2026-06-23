from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_control.env import DroneGoalEnv


def main():
    env = DroneGoalEnv(gui=False)
    obs, info = env.reset(seed=0)
    assert obs.shape == (16,), f"Unexpected obs shape: {obs.shape}"

    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4))
        assert obs.shape == (16,)
        if terminated or truncated:
            break

    env.close()
    print("Project smoke test passed.")


if __name__ == "__main__":
    main()
