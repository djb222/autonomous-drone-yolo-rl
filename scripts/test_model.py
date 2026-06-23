from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from stable_baselines3 import PPO
from drone_control.env import DroneGoalEnv


def main():
    parser = ArgumentParser(description="Run a trained PPO drone model in the Gym environment.")
    parser.add_argument("--model", default="ppo_pybullet_drone_test.zip", help="Model file name located at project root")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--debug", action="store_true", help="Enable environment debug logging")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run training first or provide the correct --model path."
        )

    env = DroneGoalEnv(gui=args.gui, render_sleep=args.gui, debug=args.debug)

    model = PPO.load(str(model_path), device="cpu")

    obs, info = env.reset(seed=args.seed)
    print(f"Starting evaluation: seed={args.seed}, goal_index={info['goal_index']}, goal={info['current_goal']}")

    terminated = False
    truncated = False
    step = 0
    total_reward = 0.0

    try:
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if step % 50 == 0 or terminated or truncated:
                print(
                    f"step={step:04d} reward={reward:.3f} total={total_reward:.3f} "
                    f"terminated={terminated} truncated={truncated} "
                    f"goal_index={info['goal_index']} dist={info['distance_to_goal']:.3f} "
                    f"reason={info.get('termination_reason')}")

        print(f"Run complete: steps={step}, total_reward={total_reward:.3f}, terminated={terminated}, truncated={truncated}")
        print(f"final_info={info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()