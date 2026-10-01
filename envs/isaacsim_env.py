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
"""Isaac Sim environment wrapper driving a task, a robot and cameras."""

import math
import os
from importlib import import_module

import numpy as np
import rootutils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

rootutils.setup_root(__file__, pythonpath=True)

from .robots import BaseRobot  # noqa: E402


class IsaacsimEnv:
    """High-level Isaac Sim environment.

    Instantiates the task class referenced in the configuration, sets up the
    simulation world, adds the robot, objects and cameras, and exposes a
    minimal ``reset`` / ``step`` / ``get_observation`` API.
    """

    def __init__(self, cfg):
        """Build the environment from a parsed YAML configuration dictionary."""
        self.cfg = cfg
        self.cameras = []
        self.task_name = None
        self.task = None
        self.world = None
        self.robot = None
        self._setup_environment()

    def init_objects(self):
        """Spawn the task environment and all task-related objects."""
        # The environment USD is placed at the world origin.
        self.task.environment.init_object(self.world, np.array([0.0, 0.0,
                                                                0.0]),
                                          np.array([1.0, 0.0, 0.0, 0.0]))

        # Resolve the robot base pose from the current configuration.
        base_position = None
        base_orientation = None

        base_name = os.environ.get('BASE_NAME', 'base_kitchen')

        if hasattr(self.robot,
                   'robot_cfg') and base_name in self.robot.robot_cfg:
            base_cfg = self.robot.robot_cfg[base_name]
            base_position = np.array(base_cfg.get('position', [0.0, 0.0, 0.0]))
            base_orientation = np.array(
                base_cfg.get('orientation', [1.0, 0.0, 0.0, 0.0]))

        for obj in self.task.objects:
            obj.init_object(self.world, base_position, base_orientation)

    def init_cameras(self):
        """Initialize all cameras defined in the configuration."""
        self.cameras = []
        camera_cfg = self.cfg.get('cameras')
        if not camera_cfg:
            raise ValueError(
                'Camera settings are missing from the configuration.')
        for _, cam in camera_cfg.items():
            camera = Camera(
                prim_path=cam['prim_path'],
                name=cam['name'],
                frequency=cam['frequency'],
                resolution=(cam['width'], cam['height']),
            )
            camera.set_local_pose(
                translation=np.array(cam['translation']),
                orientation=np.array(cam['orientation']),
                camera_axes='usd',
            )
            camera.initialize()

            fx, fy, cx, cy = cam['intrinsics']
            width, height = cam['width'], cam['height']

            # Translate pinhole intrinsics to Isaac Sim sensor parameters.
            # The sensor pixel size is fixed at 3 micrometres in this setup.
            horizontal_aperture = 3 * 1e-3 * width
            vertical_aperture = 3 * 1e-3 * height
            focal_length_x = fx * 3 * 1e-3
            focal_length_y = fy * 3 * 1e-3
            focal_length = (focal_length_x + focal_length_y) / 2

            # Isaac Sim expects focal length and apertures in units of cm.
            camera.set_focal_length(focal_length / 10.0)
            # Focus distance is in meters.
            camera.set_focus_distance(0.3)
            # f-stop is expressed in Isaac Sim units (f-number times 100).
            camera.set_lens_aperture(2.0 * 100.0)
            camera.set_horizontal_aperture(horizontal_aperture / 10.0)
            camera.set_vertical_aperture(vertical_aperture / 10.0)
            # Clipping range is in meters.
            camera.set_clipping_range(0.01, 1.0e2)

            # Configure a rational polynomial projection so that off-center
            # principal points (cx, cy) are respected. Distortion coefficients
            # are zero to keep the image undistorted.
            diagonal = 2 * math.sqrt(
                max(cx, width - cx)**2 + max(cy, height - cy)**2)
            diagonal_fov = 2 * math.atan2(diagonal, fx + fy) * 180 / math.pi
            camera.set_projection_type('fisheyePolynomial')
            camera.set_rational_polynomial_properties(width, height, cx, cy,
                                                      diagonal_fov,
                                                      np.zeros((8)))
            self.cameras.append(camera)

    def _setup_environment(self):
        """Create the world, task, robot and all scene assets."""
        self.task_name = self.cfg.get('task')

        module = import_module('tasks')
        cls = getattr(module, self.task_name)
        self.task = cls()

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 30.0,
            rendering_dt=1.0 / 30.0)

        self.robot = BaseRobot(self.cfg.get('robot'))
        self.robot.init_robot(self.world)

        self.init_objects()
        self.init_cameras()

    def reset(self):
        """Reset the simulation world, cameras, task objects and the robot."""
        self.world.reset()

        for cam in self.cameras:
            cam.post_reset()

        self.task.randomize_objects_position()

        self.robot.reset()

    def apply_action(self, action):
        """Forward an action to the underlying dual-arm robot."""
        self.robot.apply_action(action)

    def get_robot_pos(self, robot1, robot2):
        """Return the joint-position dictionary for both arms.

        Args:
            robot1: Left manipulator whose joints become ``llink1..llink8``.
            robot2: Right manipulator whose joints become ``rlink1..rlink8``.

        Returns:
            A dictionary of the form ``{'dof_pos': {<joint_name>: <pos>}}``.
        """
        dof_pos = {}
        dof_poses = (
            robot1.get_joint_positions().tolist() +
            robot2.get_joint_positions().tolist())
        for i in range(1, len(dof_poses) + 1):
            if i <= 8:
                dof_pos[f"llink{i}"] = dof_poses[i - 1]
            else:
                dof_pos[f"rlink{i - 8}"] = dof_poses[i - 1]

        return {'dof_pos': dof_pos}

    def get_observation(self):
        """Collect the current observation (camera frames, object and robot poses)."""
        obs = {}
        # Make sure the cameras have been initialized before reading frames.
        if not self.cameras:
            print('Warning: camera list is empty, reinitializing cameras...')
            self.init_cameras()

        for cam in self.cameras:
            try:
                obs[cam.name] = cam.get_current_frame()['rgba'][:, :, :3].copy(
                )
            except Exception as e:
                print(f"Failed to fetch frame from camera {cam.name}: {e}")
                # Fall back to an empty observation when frame retrieval fails.
                obs[cam.name] = None

        for obj in self.task.objects:
            obs[obj.name] = obj.get_obj_pos()

        obs[self.robot.name] = self.get_robot_pos(self.robot.left_manipulator,
                                                  self.robot.right_manipulator)
        return obs

    def step(self, action=None):
        """Apply an action (optional) and return the updated observation."""
        if action is not None:
            self.apply_action(action)
            self.world.step(render=True)

        obs = self.get_observation()
        return obs
