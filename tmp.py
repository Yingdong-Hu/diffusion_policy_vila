import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st

import torch
import numpy as np
import zerorpc
from diffusion_policy.common.pose_util import pose_to_mat, mat_to_pose


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface:
    def __init__(self, ip='172.16.0.1', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        flange_pose = np.array(self.server.get_ee_pose())
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        return tip_pose

    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())

    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )

    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()


if __name__ == "__main__":
    robot = FrankaInterface()

    # Get joint positions
    joint_positions = robot.get_joint_positions()
    print(f"Current joint positions: {joint_positions}")

    # Command robot to pose (move 4th and 6th joint)
    joint_positions_desired = np.array(
        [-0.14, -0.02, -0.05, -1.57, 0.05, 1.50, -0.91]
    )
    print(f"\nMoving joints to: {joint_positions_desired} ...\n")
    state_log = robot.move_to_joint_positions(joint_positions_desired, time_to_go=2.0)

    # Get updated joint positions
    joint_positions = robot.get_joint_positions()
    print(f"New joint positions: {joint_positions}")
