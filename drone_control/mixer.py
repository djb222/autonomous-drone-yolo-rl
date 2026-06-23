import numpy as np

from .config import DroneConfig
from .math_utils import clamp


class QuadrotorMixer:
    '''
    Converts:
        total thrust + roll/pitch/yaw torque
    into:
        front/right/rear/left motor forces.
    '''

    def __init__(self, cfg: DroneConfig):
        self.cfg = cfg
        l = cfg.arm_length
        c = cfg.yaw_torque_coeff

        self.A = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],   # total thrust
                [0.0, l, 0.0, -l],      # roll torque
                [-l, 0.0, l, 0.0],      # pitch torque
                [c, -c, c, -c],         # yaw torque
            ],
            dtype=float,
        )
        self.A_inv = np.linalg.pinv(self.A)

    def mix(self, total_thrust: float, body_torques):
        desired = np.array(
            [total_thrust, body_torques[0], body_torques[1], body_torques[2]],
            dtype=float,
        )
        forces = self.A_inv @ desired
        return clamp(forces, self.cfg.min_rotor_force, self.cfg.max_rotor_force)
