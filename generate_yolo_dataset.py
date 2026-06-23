import os
import random
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

from drone_control.env import DroneGoalEnv


WIDTH = 640
HEIGHT = 480
NUM_IMAGES = 1000

DATASET_DIR = Path("datasets/phone_dataset")
IMAGE_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


def save_yolo_label(label_path, boxes):
    with open(label_path, "w") as f:
        for box in boxes:
            class_id, x_center, y_center, w, h = box
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def get_bbox_from_mask(seg_mask, object_id):
    object_mask = (seg_mask & ((1 << 24) - 1)) == object_id

    ys, xs = np.where(object_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w < 3 or box_h < 3:
        return None

    x_center = ((x_min + x_max) / 2) / WIDTH
    y_center = ((y_min + y_max) / 2) / HEIGHT
    w = box_w / WIDTH
    h = box_h / HEIGHT

    return [0, x_center, y_center, w, h]


env = DroneGoalEnv(
    gui=False,
    enable_obstacles=True,
    render_sleep=False
)

env.reset()

for i in range(NUM_IMAGES):
    # Randomise drone/camera position
    drone_x = random.uniform(-2.0, 2.0)
    drone_y = random.uniform(-2.0, 2.0)
    drone_z = random.uniform(1.2, 2.5)

    p.resetBasePositionAndOrientation(
        env.drone.body_id,
        [drone_x, drone_y, drone_z],
        [0, 0, 0, 1],
        physicsClientId=env.client_id
    )

    # Top-down camera above drone
    camera_eye = [drone_x, drone_y, drone_z + random.uniform(3.5, 6.0)]
    camera_target = [drone_x, drone_y, 0.0]
    camera_up = [0, 1, 0]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=camera_target,
        cameraUpVector=camera_up
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=random.uniform(80, 110),
        aspect=WIDTH / HEIGHT,
        nearVal=0.1,
        farVal=20.0
    )

    img = p.getCameraImage(
        width=WIDTH,
        height=HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=env.client_id
    )

    rgba = np.reshape(img[2], (HEIGHT, WIDTH, 4)).astype(np.uint8)
    rgb = rgba[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    seg_mask = np.reshape(img[4], (HEIGHT, WIDTH))

    boxes = []

    for phone_id in env.goal_vis_ids:
        bbox = get_bbox_from_mask(seg_mask, phone_id)

        if bbox is not None:
            boxes.append(bbox)

    image_path = IMAGE_DIR / f"image_{i:05d}.jpg"
    label_path = LABEL_DIR / f"image_{i:05d}.txt"

    cv2.imwrite(str(image_path), bgr)
    save_yolo_label(label_path, boxes)

    print(f"Saved {image_path} with {len(boxes)} labels")

env.close()

print("Dataset generation complete.")