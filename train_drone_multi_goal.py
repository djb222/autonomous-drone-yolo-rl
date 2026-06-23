import os
import math
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from drone_control.env import DroneGoalEnv

def make_env():
    return DroneGoalEnv(
        gui=False,
        enable_obstacles=False,
        debug=False,
        reward_callback=custom_reward,
    )

# ========================================================
# Training Configuration
# ========================================================
ENV_ID = "DroneMultiGoal-v0"
LOAD_NAME = "Working+MovingGoalsFurtherAway8"
SAVE_NAME = "Working+MovingGoalsFurtherAway9"
TENSORBOARD_DIR = "tensorboard/"
CHECKPOINT_DIR = "checkpoints/"

TOTAL_TIMESTEPS = 250_000
N_ENVS = 4


# ========================================================
# RL Action Interface
# ========================================================
# The PPO policy should output one continuous target array each environment step:
#
# action = [x_target, y_target, z_target, yaw_target]
#
# This is a HIGH-LEVEL target command, not direct motor control.
# The environment/controller should interpret this target and convert it into
# forces, velocities, thrust, or motor commands.
#
# Data type expected by Gymnasium/SB3: np.ndarray with dtype=np.float32
# Shape: (4,)
ACTION_DIM = 4
ACTION_LABELS = ["x_target", "y_target", "z_target", "yaw_target"]

# Example target bounds for the training environment.
# These should match the physical/simulation limits of the drone world.
ACTION_LOW = np.array([-10.0, -10.0, 0.2, -math.pi], dtype=np.float32)
ACTION_HIGH = np.array([10.0, 10.0, 5.0, math.pi], dtype=np.float32)


def decode_action(action):
    """
    Convert PPO output into a readable step target dictionary.

    Args:
        action: np.ndarray/list shaped like [x_target, y_target, z_target, yaw_target].

    Returns:
        dict: Named target values, all as Python floats.
    """
    action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    action = np.clip(action, ACTION_LOW, ACTION_HIGH)

    return {
        "x_target": float(action[0]),
        "y_target": float(action[1]),
        "z_target": float(action[2]),
        "yaw_target": float(action[3]),
    }


STEP_PENALTY = -0.005
PROGRESS_REWARD_SCALE = 6.0
WAYPOINT_REWARD = 200.0
FINAL_GOAL_REWARD = 500.0

COLLISION_PENALTY = -5000.0
OUT_OF_BOUNDS_PENALTY = -300.0
TIMEOUT_PENALTY = -50.0

ACTION_PENALTY_SCALE = 0.03
CRASH_PENALTY = -1000.0
LOW_ALTITUDE_PENALTY = -10.0
MIN_SAFE_ALTITUDE = 0.45
# ========================================================
# Helper Functions
# ========================================================
def euclidean_distance(a, b):
    """Return 3D Euclidean distance between two points."""
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def safe_normalise_index(index, total_goals):
    """Normalise current goal index to 0-1 range."""
    if total_goals <= 1:
        return 0.0
    return float(index) / float(total_goals - 1)


# ========================================================
# Custom Observation Function
# ========================================================
def custom_observation(
    drone_pos,
    drone_orn,
    current_goal_pos,
    final_goal_pos,
    next_goal_pos=None,
    current_goal_index=0,
    total_goals=1,
    **kwargs,
):
    """
    Build the observation vector used by the neural network.

    This version keeps the observation RL-focused and simple.
    It assumes the environment provides global drone and goal positions.

    Args:
        drone_pos: Current drone position [x, y, z].
        drone_orn: Current drone orientation. Included for future use, not required here.
        current_goal_pos: Active target waypoint [x, y, z].
        final_goal_pos: Final waypoint in the mission [x, y, z].
        next_goal_pos: Next waypoint after the current one. If unavailable, use current goal.
        current_goal_index: Integer index of the active waypoint.
        total_goals: Total number of waypoints in the mission.
        **kwargs: Allows extra values from the environment without breaking this function.

    Returns:
        np.ndarray: Observation vector for PPO.
    """
    if next_goal_pos is None:
        next_goal_pos = current_goal_pos

    drone_pos = np.array(drone_pos, dtype=np.float32)
    current_goal_pos = np.array(current_goal_pos, dtype=np.float32)
    next_goal_pos = np.array(next_goal_pos, dtype=np.float32)
    final_goal_pos = np.array(final_goal_pos, dtype=np.float32)

    rel_current_goal = current_goal_pos - drone_pos
    rel_next_goal = next_goal_pos - drone_pos

    dist_to_current_goal = np.linalg.norm(rel_current_goal)
    dist_to_final_goal = np.linalg.norm(final_goal_pos - drone_pos)

    goal_index_norm = safe_normalise_index(current_goal_index, total_goals)

    observation = np.array([
        rel_current_goal[0],
        rel_current_goal[1],
        rel_current_goal[2],
        rel_next_goal[0],
        rel_next_goal[1],
        rel_next_goal[2],
        goal_index_norm,
        dist_to_current_goal,
        dist_to_final_goal,
    ], dtype=np.float32)

    return observation


# ========================================================
# Custom Reward Function
# ========================================================
def custom_reward(
    drone_pos,
    current_goal_pos,
    previous_distance_to_goal,
    current_distance_to_goal,
    reached_current_goal=False,
    reached_final_goal=False,
    collided=False,
    out_of_bounds=False,
    timed_out=False,
    action=None,
    termination_reason=None,
    **kwargs,
):
    """
    Compute reward for a multi-goal drone mission.

    Args:
        drone_pos: Current drone position [x, y, z].
        current_goal_pos: Active goal position [x, y, z].
        previous_distance_to_goal: Distance to active goal at previous step.
        current_distance_to_goal: Distance to active goal now.
        reached_current_goal: True if active waypoint was reached this step.
        reached_final_goal: True if final waypoint was reached this step.
        collided: True if drone collided with an obstacle.
        out_of_bounds: True if drone left allowed flight area.
        timed_out: True if episode exceeded max steps.
        action: Optional target action [x_target, y_target, z_target, yaw_target], used for small action penalty.
        **kwargs: Allows extra values from the environment without breaking this function.

    Returns:
        float: Scalar reward.
    """
    reward = 0.0


    # Make inputs robust in case lists/tuples are passed in from env.py.
    drone_pos = np.asarray(drone_pos, dtype=np.float32)
    current_goal_pos = np.asarray(current_goal_pos, dtype=np.float32)

    # Extract altitude safely. This fixes the previous NameError from using
    # drone_z without defining it first.
    drone_z = float(drone_pos[2]) if drone_pos.shape[0] >= 3 else 0.0

    # Some env implementations pass this in kwargs instead of as a named arg.
    if termination_reason is None:
        termination_reason = kwargs.get("termination_reason", None)
    if termination_reason is None:
        termination_reason = kwargs.get("terminated_reason", None)

    # Treat common crash/collision reason strings as crashes.
    reason_text = str(termination_reason).lower() if termination_reason is not None else ""
    crashed = collided or reason_text in {"crash", "collision", "obstacle_collision", "ground_collision"}

    # 1. Time/step penalty
    reward += STEP_PENALTY

    # 2. Progress toward current waypoint
    progress = previous_distance_to_goal - current_distance_to_goal
    reward += PROGRESS_REWARD_SCALE * progress

    # 3. Reward reaching intermediate waypoint
    if reached_current_goal:
        reward += WAYPOINT_REWARD

    # 4. Larger reward for completing the whole goal sequence
    if reached_final_goal:
        reward += FINAL_GOAL_REWARD

    # 5. Failure penalties
    if collided:
        reward += COLLISION_PENALTY

    if out_of_bounds:
        reward += OUT_OF_BOUNDS_PENALTY

    if timed_out:
        reward += TIMEOUT_PENALTY

    # 6. Optional action penalty for smoother target commands
    # Here the action is [x_target, y_target, z_target, yaw_target].
    if action is not None:
        action = np.array(action, dtype=np.float32)
        reward -= ACTION_PENALTY_SCALE * float(np.sum(np.square(action)))

    return float(reward)


# ========================================================
# Environment Factory
# ========================================================
def make_env_kwargs():
    """
    Build keyword arguments passed into the custom Gymnasium environment.

    Custom MultiGoalDroneEnv should accept these callbacks and call them
    inside its step/reset logic.

    The environment should define its action_space using ACTION_LOW/ACTION_HIGH:

        self.action_space = gym.spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            dtype=np.float32
        )

    Each step, the environment receives an action array:

        [x_target, y_target, z_target, yaw_target]
    """
    return {
        "renders": False,
        "reward_callback": custom_reward,
        "observation_callback": custom_observation,

        # RL-to-control interface. The environment/controller should use this
        # action format each step: [x_target, y_target, z_target, yaw_target].
        "action_low": ACTION_LOW,
        "action_high": ACTION_HIGH,
        "action_labels": ACTION_LABELS,
        "action_decoder": decode_action,

        # Example environment-level parameters. Environment can choose
        # whether to support these.
        "goal_threshold": 0.5,
        "max_episode_steps": 1000,
        "mission_goals": [
            [2.0, 1.0, 1.0],
            [4.0, 2.0, 1.5],
            [6.0, 0.0, 1.0],
        ],
    }


# ========================================================
# Main Training Loop
# ========================================================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env_kwargs = make_env_kwargs()

    # This requires environment.py to register/provide a Gymnasium-compatible
    # environment with action_space = Box(ACTION_LOW, ACTION_HIGH).
    env = make_vec_env(
    make_env,
    n_envs=N_ENVS,
    vec_env_cls=SubprocVecEnv,
    vec_env_kwargs={"start_method": "spawn"},
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix=LOAD_NAME,
    )

    if os.path.exists(f"{LOAD_NAME}.zip"):
        print(f"Loading existing model: {LOAD_NAME}.zip")
        model = PPO.load(LOAD_NAME, env=env)
    else:
        print("Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=256,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=TENSORBOARD_DIR,
        )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(SAVE_NAME)
    env.close()
    print(f"Saved model as {SAVE_NAME}.zip")


if __name__ == "__main__":
    main()
