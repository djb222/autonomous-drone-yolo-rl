import numpy as np
import pybullet as p

from .config import DroneConfig
from .math_utils import rotation_matrix_from_quaternion


class Quadrotor:
    def __init__(self, cfg: DroneConfig, client_id: int):
        self.cfg = cfg
        self.client_id = client_id
        self.body_id = None

        l = cfg.arm_length
        self.rotor_positions_body = np.array(
            [
                [l, 0.0, 0.0],    # front
                [0.0, l, 0.0],    # right
                [-l, 0.0, 0.0],   # rear
                [0.0, -l, 0.0],   # left
            ],
            dtype=float,
        )
        self.spin_dirs = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)

    def load(self, position=(0, 0, 1), orientation=(0, 0, 0, 1)):
        self.body_id = p.loadURDF(
            str(self.cfg.urdf_path),
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            self.body_id,
            -1,
            mass=self.cfg.mass,
            linearDamping=0.15,
            angularDamping=0.20,
            physicsClientId=self.client_id,
        )
        return self.body_id

    def reset(self, position=(0, 0, 1), orientation=(0, 0, 0, 1)):
        if self.body_id is None:
            return self.load(position, orientation)

        p.resetBasePositionAndOrientation(
            self.body_id, position, orientation, physicsClientId=self.client_id
        )
        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=(0, 0, 0),
            angularVelocity=(0, 0, 0),
            physicsClientId=self.client_id,
        )
        return self.body_id

    def get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        vel, ang_vel = p.getBaseVelocity(self.body_id, physicsClientId=self.client_id)
        euler = p.getEulerFromQuaternion(quat)

        return {
            "pos": np.array(pos, dtype=float),
            "quat": np.array(quat, dtype=float),
            "euler": np.array(euler, dtype=float),
            "vel": np.array(vel, dtype=float),
            "ang_vel": np.array(ang_vel, dtype=float),
            "rot": rotation_matrix_from_quaternion(quat),
        }

    def apply_motor_forces(self, motor_forces):
        state = self.get_state()
        pos = state["pos"]
        rot = state["rot"]
        motor_forces = np.asarray(motor_forces, dtype=float)

        for force, rotor_body in zip(motor_forces, self.rotor_positions_body):
            world_point = pos + rot @ rotor_body
            world_force = rot @ np.array([0.0, 0.0, force], dtype=float)

            p.applyExternalForce(
                self.body_id,
                -1,
                forceObj=world_force.tolist(),
                posObj=world_point.tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client_id,
            )

        yaw_torque_body = np.array(
            [0.0, 0.0, self.cfg.yaw_torque_coeff * float(np.dot(self.spin_dirs, motor_forces))],
            dtype=float,
        )
        yaw_torque_world = rot @ yaw_torque_body

        p.applyExternalTorque(
            self.body_id,
            -1,
            torqueObj=yaw_torque_world.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.client_id,
        )
