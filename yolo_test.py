import time
import cv2
from stable_baselines3 import PPO

from drone_control.env import DroneGoalEnv

MODEL_PATH = "3GoalsV6DroneDirectionChangesV4.zip"
YOLO_MODEL_PATH = "./yolo_phone_best.pt"

ppo_model = PPO.load(MODEL_PATH)

env = DroneGoalEnv(
    gui=True,
    enable_obstacles=True,
    render_sleep=False,
    enable_yolo=True,
    yolo_model_path=YOLO_MODEL_PATH,
    yolo_conf_threshold=0.25,
    yolo_detection_interval=5,
    require_yolo_for_goal=False,
)

obs, info = env.reset()

try:
    while True:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("yolo_detected"):
            print(
                f"YOLO detected {info.get('yolo_class_name')} "
                f"conf={info.get('yolo_confidence'):.2f} "
                f"centered={info.get('yolo_centered')} "
                f"offset={info.get('yolo_offset')} "
                f"goal_index={info.get('goal_index')} "
                f"reward={reward:.2f}"
            )
        else:
            print(
                f"No YOLO detection | "
                f"goal_index={info.get('goal_index')} "
                f"dist={info.get('distance_to_goal'):.2f} "
                f"reward={reward:.2f}"
            )

        annotated_frame = env.get_last_yolo_frame()
        if annotated_frame is not None:
            cv2.imshow("YOLO Detection", annotated_frame)

        if terminated or truncated:
            print(f"Environment reset | reason={info.get('termination_reason')}")
            obs, info = env.reset()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(1 / 60)

except KeyboardInterrupt:
    pass

finally:
    env.close()
    cv2.destroyAllWindows()