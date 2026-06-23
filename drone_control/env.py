import time
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .config import DroneConfig
from .pid import CascadedPIDController
from .quadrotor import Quadrotor

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = object
    spaces = None


class DroneGoalEnv(gym.Env if hasattr(gym, "Env") else object):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        goals=None,
        obstacles=None,
        enable_obstacles=False,
        gui=False,
        cfg=None,
        render_sleep=False,
        debug=False,
        reward_callback=None,
        enable_yolo=False,
        yolo_model_path="./yolo_phone_best.pt",
        yolo_conf_threshold=0.25,
        yolo_camera_width=640,
        yolo_camera_height=480,
        yolo_detection_interval=5,
        yolo_center_tolerance_px=120.0,
        require_yolo_for_goal=False,
        yolo_reward_bonus=25.0,
        yolo_centered_reward_bonus=50.0,
    ):
        self.cfg = cfg or DroneConfig()
        self.gui = bool(gui)
        self.render_sleep = bool(render_sleep)
        self.debug = bool(debug)
        self.enable_obstacles = bool(enable_obstacles)
        self.obstacle_data = obstacles
        self.reward_callback = reward_callback

        
        # YOLO is used internally for reward/goal-confirmation/info only.
        self.enable_yolo = bool(enable_yolo)
        self.yolo_model_path = yolo_model_path
        self.yolo_conf_threshold = float(yolo_conf_threshold)
        self.yolo_camera_width = int(yolo_camera_width)
        self.yolo_camera_height = int(yolo_camera_height)
        self.yolo_detection_interval = max(1, int(yolo_detection_interval))
        self.yolo_center_tolerance_px = float(yolo_center_tolerance_px)
        self.require_yolo_for_goal = bool(require_yolo_for_goal)
        self.yolo_reward_bonus = float(yolo_reward_bonus)
        self.yolo_centered_reward_bonus = float(yolo_centered_reward_bonus)
        self.yolo_model = None
        self.last_yolo_result = {
            "detected": False,
            "centered": False,
            "confidence": 0.0,
            "class_name": None,
            "center": None,
            "offset": None,
            "annotated_frame": None,
        }

        if self.enable_yolo:
            if YOLO is None:
                raise ImportError("enable_yolo=True requires ultralytics to be installed")
            if cv2 is None:
                raise ImportError("enable_yolo=True requires opencv-python/cv2 to be installed")
            self.yolo_model = YOLO(self.yolo_model_path)

        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        self.project_root = Path(__file__).resolve().parents[1]
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(str(self.project_root))
        p.setAdditionalSearchPath(str(self.cfg.asset_dir))

        self.goals = np.asarray(
            goals if goals is not None else [
                # Stage 1 training goals: fly above each phone/table rather than into them.
                [1.2, 0.4, 1.2],
                [1.6, 0.6, 1.2],
                [1.4, 1.0, 1.2],
            ],
            dtype=float,
        )

        self.drone = None
        self.controller = CascadedPIDController(self.cfg)
        self.goal_index = 0
        self.step_count = 0
        self.target_yaw = 0.0
        self.last_action = np.zeros(4, dtype=float)
        self.obstacle_ids = []
        self.goal_vis_ids = []
        self.previous_distance_to_goal = None

        if spaces is not None:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
            )

        self._build_world()

    def _build_world(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(str(self.project_root))
        p.setAdditionalSearchPath(str(self.cfg.asset_dir))
        p.setGravity(0, 0, -self.cfg.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.cfg.dt, physicsClientId=self.client_id)
        p.loadURDF(str(pybullet_data.getDataPath()) + "/plane.urdf", physicsClientId=self.client_id)

        self.drone = Quadrotor(self.cfg, self.client_id)
        self.drone.load(position=(0, 0, 1))
        if self.enable_obstacles:
            self._create_obstacles()
        self._create_goal_objects()
        self._draw_goals()

    def get_goal_positions(self):
        return self.goals.copy()


    def get_phone_ids(self):
        return list(self.goal_vis_ids)


    def get_goal_for_phone_id(self, phone_id):
        if phone_id not in self.goal_vis_ids:
            return None

        index = self.goal_vis_ids.index(phone_id)
        return self.goals[index].copy(), index    

    def _draw_goals(self):
        if not self.gui:
            return

        for goal in self.goals:
            p.addUserDebugLine(goal + np.array([-0.15, 0, 0]), goal + np.array([0.15, 0, 0]), [0, 1, 0], 2, physicsClientId=self.client_id)
            p.addUserDebugLine(goal + np.array([0, -0.15, 0]), goal + np.array([0, 0.15, 0]), [0, 1, 0], 2, physicsClientId=self.client_id)
            p.addUserDebugLine(goal + np.array([0, 0, -0.15]), goal + np.array([0, 0, 0.15]), [0, 1, 0], 2, physicsClientId=self.client_id)

    def _create_obstacles(self):
        self.obstacle_ids = []
        obstacle_data = self.obstacle_data if self.obstacle_data is not None else [
            ([0.8, -0.8, 0.25], [0.2, 0.6, 0.25]),
            ([-0.8, 0.8, 0.25], [0.2, 0.7, 0.25]),
            ([0.0, 0.4, 0.25], [0.7, 0.2, 0.25]),
        ]

        for position, half_extents in obstacle_data:
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.5, 0.5, 0.5, 1],
                physicsClientId=self.client_id,
            )
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=position,
                physicsClientId=self.client_id,
            )
            self.obstacle_ids.append(obstacle_id)

    def _create_goal_objects(self):
        self.goal_vis_ids = []
        table_urdf = str(Path(pybullet_data.getDataPath()) / "table" / "table.urdf")
        phone_height = 0.65

        for goal in self.goals:
            p.loadURDF(
                table_urdf,
                basePosition=[goal[0], goal[1], 0],
                physicsClientId=self.client_id,
            )

            phone_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.18, 0.09, 0.02],
                physicsClientId=self.client_id,
            )
            phone_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.18, 0.09, 0.02],
                rgbaColor=[0.02, 0.02, 0.02, 1],
                physicsClientId=self.client_id,
            )
            phone_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=phone_collision,
                baseVisualShapeIndex=phone_visual,
                basePosition=[goal[0], goal[1], phone_height],
                physicsClientId=self.client_id,
            )
            self.goal_vis_ids.append(phone_id)


    def _capture_drone_camera_bgr(self):
        """Capture a downward-facing PyBullet camera image from above the drone."""
        state = self.drone.get_state()
        drone_pos = state["pos"]

        camera_eye = [drone_pos[0], drone_pos[1], drone_pos[2] + 6.0]
        camera_target = [drone_pos[0], drone_pos[1], 0.0]
        camera_up = [0, 1, 0]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_eye,
            cameraTargetPosition=camera_target,
            cameraUpVector=camera_up,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=100,
            aspect=self.yolo_camera_width / self.yolo_camera_height,
            nearVal=0.1,
            farVal=20.0,
        )
        img = p.getCameraImage(
            width=self.yolo_camera_width,
            height=self.yolo_camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            physicsClientId=self.client_id,
        )

        rgba = np.reshape(img[2], (self.yolo_camera_height, self.yolo_camera_width, 4)).astype(np.uint8)
        rgb = rgba[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _run_yolo_detection(self):
        """
        Run YOLO and store detection details.

        Important: this method does not modify the observation vector, so old PPO
        models with observation shape (16,) can still be loaded.
        """
        if not self.enable_yolo or self.yolo_model is None:
            self.last_yolo_result = {
                "detected": False,
                "centered": False,
                "confidence": 0.0,
                "class_name": None,
                "center": None,
                "offset": None,
                "annotated_frame": None,
            }
            return self.last_yolo_result

        bgr = self._capture_drone_camera_bgr()
        results = self.yolo_model.predict(
            source=bgr,
            conf=self.yolo_conf_threshold,
            verbose=False,
        )

        boxes = results[0].boxes
        if len(boxes) == 0:
            self.last_yolo_result = {
                "detected": False,
                "centered": False,
                "confidence": 0.0,
                "class_name": None,
                "center": None,
                "offset": None,
                "annotated_frame": results[0].plot(),
            }
            return self.last_yolo_result

        # Use the highest-confidence detection.
        best_box = max(boxes, key=lambda box: float(box.conf[0]))
        class_id = int(best_box.cls[0])
        confidence = float(best_box.conf[0])
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        offset_x = center_x - (self.yolo_camera_width / 2.0)
        offset_y = center_y - (self.yolo_camera_height / 2.0)
        centered = abs(offset_x) <= self.yolo_center_tolerance_px and abs(offset_y) <= self.yolo_center_tolerance_px

        self.last_yolo_result = {
            "detected": True,
            "centered": bool(centered),
            "confidence": confidence,
            "class_name": self.yolo_model.names.get(class_id, str(class_id)),
            "center": (float(center_x), float(center_y)),
            "offset": (float(offset_x), float(offset_y)),
            "annotated_frame": results[0].plot(),
        }
        return self.last_yolo_result

    def get_last_yolo_frame(self):
        """Return the latest annotated YOLO frame for display in yolo_test.py."""
        return self.last_yolo_result.get("annotated_frame")

    def _check_obstacle_collision(self):
        if not self.enable_obstacles:
            return False

        for obstacle_id in self.obstacle_ids:
            contacts = p.getContactPoints(
                bodyA=self.drone.body_id,
                bodyB=obstacle_id,
                physicsClientId=self.client_id,
            )
            if len(contacts) > 0:
                return True
        return False

    def _debug(self, message):
        if self.debug:
            print(f"[DroneGoalEnv DEBUG] {message}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self._build_world()
        self.goal_index = 0
        self.step_count = 0
        self.target_yaw = 0.0
        self.last_action = np.zeros(4, dtype=float)
        self.controller.reset()
        self.previous_distance_to_goal = self._distance_to_current_goal()
        self.last_yolo_result = {
            "detected": False,
            "centered": False,
            "confidence": 0.0,
            "class_name": None,
            "center": None,
            "offset": None,
            "annotated_frame": None,
        }

        self._debug(f"reset: goal_index=0, goal={self.goals[0].tolist()}, max_steps={self.cfg.max_episode_steps}")

        obs = self._get_obs()
        return obs, {"goal_index": self.goal_index, "current_goal": self.goals[self.goal_index].copy()}

    def _get_obs(self):
        s = self.drone.get_state()
        current_goal = self.goals[min(self.goal_index, len(self.goals) - 1)]
        rel_goal = current_goal - s["pos"]
        goal_fraction = np.array([self.goal_index / max(1, len(self.goals) - 1)], dtype=float)

        

        return np.concatenate(
            [s["pos"], s["vel"], s["euler"], s["ang_vel"], rel_goal, goal_fraction]
        ).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(4)
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.copy()

        # During early training, PPO controls horizontal velocity and yaw only.
        # Altitude is held by the PID controller at a fixed safe target height.
        vel_cmd = action[:3] * self.cfg.max_vel_cmd
        vel_cmd[2] = 0.0
        yaw_rate_cmd = action[3] * self.cfg.max_yaw_rate_cmd

        for _ in range(self.cfg.action_repeat):
            state = self.drone.get_state()

            lookahead_time = 0.30
            target_pos = state["pos"] + vel_cmd * lookahead_time
            target_pos[2] = 1.2

            self.target_yaw += yaw_rate_cmd * self.cfg.dt

            motor_forces = self.controller.compute_motor_forces(
                state=state,
                target_pos=target_pos,
                target_yaw=self.target_yaw,
                dt=self.cfg.dt,
            )
            self.drone.apply_motor_forces(motor_forces)
            p.stepSimulation(physicsClientId=self.client_id)

            if self.gui and self.render_sleep:
                time.sleep(self.cfg.dt)

        self.step_count += 1
        state = self.drone.get_state()
        obs = self._get_obs()

        current_distance = self._distance_to_current_goal()

        # Run YOLO at a reduced frequency to avoid making every env step too slow.
        if self.enable_yolo and (self.step_count % self.yolo_detection_interval == 0):
            yolo_result = self._run_yolo_detection()
        else:
            yolo_result = self.last_yolo_result

        close_enough_to_goal = current_distance < max(self.cfg.goal_radius, 0.40)
        yolo_confirmed = bool(yolo_result.get("detected", False))

        # If require_yolo_for_goal=True, the drone must be close enough AND YOLO must
        # see the phone/object before the waypoint counts as reached.
        # If False, old distance-only behaviour is preserved.
        reached = close_enough_to_goal and (yolo_confirmed or not self.require_yolo_for_goal)

        truncated = self.step_count >= self.cfg.max_episode_steps
        obstacle_collision = self._check_obstacle_collision()
        out_of_bounds = bool(np.any(np.abs(state["pos"]) > self.cfg.world_limit))
        crashed = bool(state["pos"][2] < 0.05 or out_of_bounds or obstacle_collision)
        reached_final_goal = bool(reached and self.goal_index == len(self.goals) - 1)

        terminated = bool(crashed or reached_final_goal)

        termination_reason = None
        if obstacle_collision:
            termination_reason = "obstacle_collision"
        elif crashed and not obstacle_collision:
            termination_reason = "crash"
        elif reached_final_goal:
            termination_reason = "goal_complete"
        elif truncated:
            termination_reason = "timeout"

        # Use the train-file reward function if one was provided. Otherwise, fall back
        # to the simple built-in reward.
        if self.reward_callback is not None:
            reward = self.reward_callback(
                drone_pos=state["pos"],
                current_goal_pos=self.goals[self.goal_index],
                previous_distance_to_goal=self.previous_distance_to_goal,
                current_distance_to_goal=current_distance,
                reached_current_goal=reached,
                reached_final_goal=reached_final_goal,
                collided=obstacle_collision,
                out_of_bounds=out_of_bounds,
                timed_out=truncated,
                action=action,
                termination_reason=termination_reason,
            )
        else:
            reward, _ = self._compute_reward(action)
            if crashed:
                reward -= 25.0
            if reached:
                reward += 10.0

        # Add YOLO-based reward without changing the observation vector.
        # This is useful during retraining; during evaluation it simply affects logged reward.
        if self.enable_yolo and yolo_result.get("detected", False):
            reward += self.yolo_reward_bonus
            if yolo_result.get("centered", False):
                reward += self.yolo_centered_reward_bonus

        # Move to the next goal after computing reward for the goal that was just reached.
        if reached and not reached_final_goal and not crashed:
            self.goal_index += 1
            self.previous_distance_to_goal = self._distance_to_current_goal()
        else:
            self.previous_distance_to_goal = current_distance

        info = {
            "goal_index": self.goal_index,
            "reached_goal": reached,
            "distance_to_goal": self._distance_to_current_goal(),
            "last_action": action.copy(),
            "obstacle_collision": obstacle_collision,
            "out_of_bounds": out_of_bounds,
            "termination_reason": termination_reason,
            "close_enough_to_goal": close_enough_to_goal,
            "yolo_detected": yolo_result.get("detected", False),
            "yolo_centered": yolo_result.get("centered", False),
            "yolo_confidence": yolo_result.get("confidence", 0.0),
            "yolo_class_name": yolo_result.get("class_name", None),
            "yolo_offset": yolo_result.get("offset", None),
        }

        if self.debug and (terminated or truncated or reached or crashed):
            self._debug(
                f"step={self.step_count} pos={state['pos']} reward={reward:.3f} "
                f"dist={self._distance_to_current_goal():.3f} reached={reached} crashed={crashed} "
                f"obstacle_collision={obstacle_collision} terminated={terminated} truncated={truncated} "
                f"reason={termination_reason}"
            )

        return obs, float(reward), bool(terminated), bool(truncated), info

    

    def _distance_to_current_goal(self):
        if self.goal_index >= len(self.goals):
            return 0.0

        pos = self.drone.get_state()["pos"]
        goal = self.goals[self.goal_index]

        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]

        return float(np.sqrt(dx**2 + dy**2))


    def _compute_reward(self, action):
        dist = self._distance_to_current_goal()

        # XY-only goal reaching for now
        reached = dist < 0.40

        action_penalty = 0.02 * float(np.linalg.norm(action))
        living_penalty = 0.01
        reward = -dist - action_penalty - living_penalty

        euler = self.drone.get_state()["euler"]
        reward -= 0.08 * (abs(euler[0]) + abs(euler[1]))

        return reward, reached

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
