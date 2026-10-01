# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dual-arm robot wrapper built on top of Isaac Sim manipulators."""

import os

import numpy as np
import yaml
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from scipy.spatial.transform import Rotation as R


class BaseRobot:
    """Dual-arm robot composed of a base prim and two :class:`SingleManipulator`.

    The robot configuration is read from ``cfg/robots.yaml``. Each arm owns a
    parallel gripper with a closed and opened joint configuration.
    """

    def __init__(self, name):
        """Initialize the robot descriptor.

        Args:
            name: Robot identifier used as a prim-path prefix (e.g. ``"piper"``).
        """
        self.robot_cfg = self._load_robot_config()
        self.name = name
        self.base_name = os.environ.get('BASE_NAME', 'base_kitchen')
        self.left_manipulator = None
        self.right_manipulator = None

    def _load_robot_config(self):
        """Load and return the full robot configuration YAML."""
        cfg_path = os.path.join(
            os.path.dirname(__file__), 'cfg', 'robots.yaml')
        with open(cfg_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config

    def init_robot(self, world):
        """Spawn the robot base and both manipulators into ``world.scene``."""
        if self.robot_cfg is None:
            raise ValueError(f"Configuration for '{self.name}' was not found.")

        # Add the robot base prim.
        if self.base_name in self.robot_cfg:
            base_cfg = self.robot_cfg[self.base_name]
            add_reference_to_stage(
                usd_path=base_cfg['usd_path'],
                prim_path=f"/World/{self.name}_base")
            base_position = np.array(base_cfg.get('position', [0.0, 0.0, 0.0]))
            base_orientation = np.array(
                base_cfg.get('orientation', [1.0, 0.0, 0.0, 0.0]))
            _ = XFormPrim(
                prim_paths_expr=f"/World/{self.name}_base",
                positions=np.array([base_position]),
                orientations=np.array([base_orientation]),
                visibilities=np.array([[0.32809, 0.37396, 0.69]]),
            )

        if 'left_arm' in self.robot_cfg:
            left_arm_cfg = self.robot_cfg['left_arm']
            add_reference_to_stage(
                usd_path=left_arm_cfg['usd_path'],
                prim_path=f"/World/{self.name}_left_arm",
            )
            # Arm pose in the config is given relative to the robot base.
            left_arm_relative_pos = np.array(left_arm_cfg['position'])
            left_arm_relative_quat = np.array(left_arm_cfg['orientation'])

            # Convert base quaternion from [w, x, y, z] to scipy's [x, y, z, w].
            base_rot = R.from_quat(base_orientation[[1, 2, 3, 0]])
            left_arm_world_pos = base_position + base_rot.apply(
                left_arm_relative_pos)
            # Rotate relative orientation and convert back to [w, x, y, z].
            left_arm_world_quat = (base_rot * R.from_quat(
                left_arm_relative_quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

            _ = XFormPrim(
                prim_paths_expr=f"/World/{self.name}_left_arm",
                positions=np.array([left_arm_world_pos]),
                orientations=np.array([left_arm_world_quat]),
                visibilities=np.array([[0.32809, 0.37396, 0.69]]),
            )
            left_eef_path = f"/World/{self.name}_left_arm/gripper_base"
            left_griper = ParallelGripper(
                end_effector_prim_path=left_eef_path,
                joint_prim_names=['joint7', 'joint8'],
                joint_opened_positions=np.array([0.035, -0.035]),
                joint_closed_positions=np.array([0.0, -0.0]),
                action_deltas=np.array([0.035, -0.035]),
            )
            self.left_manipulator = SingleManipulator(
                prim_path=f"/World/{self.name}_left_arm",
                name=f"{self.name}_left_arm",
                end_effector_prim_name='gripper_base',
                gripper=left_griper,
            )
            joints_default_positions = np.zeros(8)
            joints_default_positions[6] = 0.035
            joints_default_positions[7] = -0.035
            self.left_manipulator.set_joints_default_state(
                positions=joints_default_positions)

        if 'right_arm' in self.robot_cfg:
            right_arm_cfg = self.robot_cfg['right_arm']
            add_reference_to_stage(
                usd_path=right_arm_cfg['usd_path'],
                prim_path=f"/World/{self.name}_right_arm",
            )
            right_arm_relative_pos = np.array(right_arm_cfg['position'])
            right_arm_relative_quat = np.array(right_arm_cfg['orientation'])

            base_rot = R.from_quat(base_orientation[[1, 2, 3, 0]])
            right_arm_world_pos = base_position + base_rot.apply(
                right_arm_relative_pos)
            right_arm_world_quat = (base_rot * R.from_quat(
                right_arm_relative_quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

            _ = XFormPrim(
                prim_paths_expr=f"/World/{self.name}_right_arm",
                positions=np.array([right_arm_world_pos]),
                orientations=np.array([right_arm_world_quat]),
                visibilities=np.array([[0.32809, 0.37396, 0.69]]),
            )
            right_eef_path = f"/World/{self.name}_right_arm/gripper_base"
            right_griper = ParallelGripper(
                end_effector_prim_path=right_eef_path,
                joint_prim_names=['joint7', 'joint8'],
                joint_opened_positions=np.array([0.035, -0.035]),
                joint_closed_positions=np.array([0.0, -0.0]),
                action_deltas=np.array([0.035, -0.035]),
            )
            self.right_manipulator = SingleManipulator(
                prim_path=f"/World/{self.name}_right_arm",
                name=f"{self.name}_right_arm",
                end_effector_prim_name='gripper_base',
                gripper=right_griper,
            )
            joints_default_positions = np.zeros(8)
            joints_default_positions[6] = 0
            joints_default_positions[7] = 0
            self.right_manipulator.set_joints_default_state(
                positions=joints_default_positions)

        world.scene.add(self.left_manipulator)
        world.scene.add(self.right_manipulator)

    def reset(self):
        """Reset both arms to a randomized home pose with an opened gripper."""
        noise = np.random.uniform(-0.1, 0.1, 6)
        left_joints = np.array(
            [-0.2, 0.3491, -0.3491, 0.0, 0.1, 0.0, 0.035, -0.035])
        left_joints[:6] += noise
        self.left_manipulator.set_joints_default_state(positions=left_joints)
        self.left_manipulator.post_reset()
        self.left_manipulator.gripper.set_joint_positions(
            self.left_manipulator.gripper.joint_opened_positions)

        right_joints = np.array(
            [0.2, 0.3491, -0.3491, 0.0, 0.1, 0.0, 0.035, -0.035])
        right_joints[:6] += noise
        self.right_manipulator.set_joints_default_state(positions=right_joints)
        self.right_manipulator.post_reset()
        self.right_manipulator.gripper.set_joint_positions(
            self.right_manipulator.gripper.joint_opened_positions)

    def apply_action(self, action):
        """Apply the given actions to the left and right manipulators.

        Args:
            action: A sequence of length 1 or 2. A length-2 sequence applies
                ``action[0]`` to the left arm and ``action[1]`` to the right
                arm (either entry may be ``None`` to skip that arm). A
                length-1 sequence applies ``action[0]`` to the right arm only.
        """
        if len(action) > 1:
            if action[0] is not None:
                self.left_manipulator.apply_action(action[0])
            if action[1] is not None:
                self.right_manipulator.apply_action(action[1])
        else:
            self.right_manipulator.apply_action(action[0])
