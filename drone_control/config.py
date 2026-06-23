from dataclasses import dataclass
from pathlib import Path


@dataclass
class DroneConfig:
    gravity: float = 9.81
    physics_hz: int = 240
    control_hz: int = 60

    mass: float = 1.0
    arm_length: float = 0.18
    max_rotor_force: float = 5.5
    min_rotor_force: float = 0.0
    yaw_torque_coeff: float = 0.010

    # Conservative gains for the simple PyBullet rigid-body drone.
    kp_pos_xy: float = 1.2
    kp_pos_z: float = 2.4
    kd_pos_xy: float = 1.3
    kd_pos_z: float = 2.0
    ki_pos_z: float = 0.02

    kp_roll_pitch: float = 2.2
    kd_roll_pitch: float = 0.55
    kp_yaw: float = 1.0
    kd_yaw: float = 0.25

    max_tilt_rad: float = 0.28
    max_xy_accel: float = 2.0
    max_z_accel: float = 3.0
    max_vel_cmd: float = 1.0
    max_yaw_rate_cmd: float = 0.8

    # Important safety limits.
    max_roll_pitch_torque: float = 0.18
    max_yaw_torque: float = 0.08

    goal_radius: float = 0.25
    max_episode_steps: int = 1500
    world_limit: float = 8.0

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def action_repeat(self) -> int:
        return max(1, int(self.physics_hz / self.control_hz))

    @property
    def hover_force_per_rotor(self) -> float:
        return self.mass * self.gravity / 4.0

    @property
    def asset_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "assets"

    @property
    def urdf_path(self) -> Path:
        return self.asset_dir / "quadrotor.urdf"
