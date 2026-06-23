import math
import numpy as np

from .config import DroneConfig
from .math_utils import clamp, wrap_angle
from .mixer import QuadrotorMixer


class PID:
    def __init__(self, kp, ki, kd, integral_limit=2.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, error, derivative_measurement, dt):
        self.integral += float(error) * dt
        self.integral = float(clamp(self.integral, -self.integral_limit, self.integral_limit))
        return self.kp * error + self.ki * self.integral - self.kd * derivative_measurement


class CascadedPIDController:
    '''
    Safer cascaded controller.

    Input:
        state, target_pos, target_yaw

    Output:
        four motor forces

    Internal flow:
        position error
        -> desired acceleration
        -> desired roll/pitch/yaw
        -> limited torques
        -> motor mixer
    '''

    def __init__(self, cfg: DroneConfig):
        self.cfg = cfg
        self.mixer = QuadrotorMixer(cfg)
        self.z_integrator = PID(0.0, cfg.ki_pos_z, 0.0, integral_limit=1.0)
        self.target_yaw = 0.0

    def reset(self):
        self.z_integrator.reset()
        self.target_yaw = 0.0

    def compute_motor_forces(self, state, target_pos, target_yaw=None, dt=None):
        cfg = self.cfg
        dt = cfg.dt if dt is None else float(dt)

        pos = state["pos"]
        vel = state["vel"]
        euler = state["euler"]
        ang_vel = state["ang_vel"]

        if target_yaw is None:
            target_yaw = self.target_yaw
        self.target_yaw = float(target_yaw)

        target_pos = np.asarray(target_pos, dtype=float)
        error_pos = target_pos - pos

        # Position controller.
        ax = cfg.kp_pos_xy * error_pos[0] - cfg.kd_pos_xy * vel[0]
        ay = cfg.kp_pos_xy * error_pos[1] - cfg.kd_pos_xy * vel[1]
        az = cfg.kp_pos_z * error_pos[2] - cfg.kd_pos_z * vel[2]
        az += self.z_integrator.update(error_pos[2], 0.0, dt)

        ax = float(clamp(ax, -cfg.max_xy_accel, cfg.max_xy_accel))
        ay = float(clamp(ay, -cfg.max_xy_accel, cfg.max_xy_accel))
        az = float(clamp(az, -cfg.max_z_accel, cfg.max_z_accel))

        # Convert desired horizontal acceleration to desired attitude.
        yaw = float(euler[2])
        desired_roll = (ax * math.sin(yaw) - ay * math.cos(yaw)) / cfg.gravity
        desired_pitch = (ax * math.cos(yaw) + ay * math.sin(yaw)) / cfg.gravity

        desired_roll = float(clamp(desired_roll, -cfg.max_tilt_rad, cfg.max_tilt_rad))
        desired_pitch = float(clamp(desired_pitch, -cfg.max_tilt_rad, cfg.max_tilt_rad))

        roll_error = wrap_angle(desired_roll - float(euler[0]))
        pitch_error = wrap_angle(desired_pitch - float(euler[1]))
        yaw_error = wrap_angle(float(target_yaw) - float(euler[2]))

        tau_x = cfg.kp_roll_pitch * roll_error - cfg.kd_roll_pitch * float(ang_vel[0])
        tau_y = cfg.kp_roll_pitch * pitch_error - cfg.kd_roll_pitch * float(ang_vel[1])
        tau_z = cfg.kp_yaw * yaw_error - cfg.kd_yaw * float(ang_vel[2])

        # Critical fix: prevent the simple drone from flipping violently.
        tau_x = float(clamp(tau_x, -cfg.max_roll_pitch_torque, cfg.max_roll_pitch_torque))
        tau_y = float(clamp(tau_y, -cfg.max_roll_pitch_torque, cfg.max_roll_pitch_torque))
        tau_z = float(clamp(tau_z, -cfg.max_yaw_torque, cfg.max_yaw_torque))

        # Collective thrust with safe tilt compensation.
        roll, pitch = float(euler[0]), float(euler[1])
        tilt_comp = max(0.75, math.cos(roll) * math.cos(pitch))
        total_thrust = cfg.mass * (cfg.gravity + az) / tilt_comp
        total_thrust = float(clamp(total_thrust, 0.0, 4.0 * cfg.max_rotor_force))

        return self.mixer.mix(total_thrust, np.array([tau_x, tau_y, tau_z], dtype=float))
