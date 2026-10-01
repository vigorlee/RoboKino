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
"""Scene-object wrappers backed by USD references."""

import os

import numpy as np
import yaml
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from scipy.spatial.transform import Rotation as R


class BaseObject:
    """Generic scene object loaded from a USD file defined in ``objects.yaml``.

    The object pose may be supplied in the robot-base frame or the world
    frame. If a robot base pose is provided at ``init_object`` time, the
    configured pose is transformed into the world frame automatically.
    """

    def __init__(self,
                 name,
                 init_position=None,
                 init_orientation=None,
                 usd_path=None):
        """Construct the object descriptor from ``objects.yaml``.

        Args:
            name: Object identifier used both as YAML key and stage prim name.
            init_position: Optional override for the initial position.
            init_orientation: Optional override for the initial orientation.
            usd_path: Currently unused; kept for API compatibility.
        """
        self.name = name
        self.cfg = self._load_object_config()
        self.usd_path = self._resolve_usd_path(self.cfg['usd_path'])
        self.init_position = (
            self.cfg['position'] if init_position is None else init_position)
        self.init_orientation = (
            self.cfg['orientation']
            if init_orientation is None else init_orientation)
        self.init_poisition_global = None
        self.object = None

    def _load_object_config(self):
        """Load and return this object's configuration block from ``objects.yaml``."""
        cfg_path = os.path.join(
            os.path.dirname(__file__), 'cfg', 'objects.yaml')
        with open(cfg_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config[self.name]

    def _resolve_usd_path(self, relative_path):
        """Resolve a USD path from the config into an absolute path."""
        cfg_dir = os.path.dirname(__file__)
        base_dir = os.path.dirname(cfg_dir)

        absolute_path = os.path.abspath(os.path.join(base_dir, relative_path))

        if not os.path.exists(absolute_path):
            raise FileNotFoundError(
                f"USD file does not exist: {absolute_path}")

        return absolute_path

    def init_object(self, world, base_position=None, base_orientation=None):
        """Add the object to the scene as an :class:`XFormPrim`.

        Args:
            world: The Isaac Sim world instance.
            base_position: Robot base position ``[x, y, z]`` in the world frame,
                or ``None`` when the configured pose is already world-frame.
            base_orientation: Robot base orientation quaternion ``[w, x, y, z]``,
                or ``None`` when the configured pose is already world-frame.
        """
        if self.object is not None:
            raise ValueError(
                f"Object '{self.name}' has already been initialized.")

        add_reference_to_stage(
            usd_path=self.usd_path, prim_path=f"/World/{self.name}")

        if base_position is not None and base_orientation is not None:
            # Configured pose is relative to the robot base; transform it.
            obj_relative_pos = np.array(self.init_position)
            obj_relative_quat = np.array(self.init_orientation)

            # Convert base quaternion from [w, x, y, z] to scipy's [x, y, z, w].
            base_rot = R.from_quat(base_orientation[[1, 2, 3, 0]])

            obj_world_pos = base_position + base_rot.apply(obj_relative_pos)
            self.init_poisition_global = obj_world_pos
            # Rotate the relative quaternion and convert back to [w, x, y, z].
            obj_world_quat = (base_rot * R.from_quat(
                obj_relative_quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

            positions = np.array([obj_world_pos])
            orientations = np.array([obj_world_quat])
        else:
            # No base pose provided; treat the configured pose as world-frame.
            self.init_poisition_global = self.init_position
            positions = np.array([self.init_position])
            orientations = np.array([self.init_orientation])

        self.object = XFormPrim(
            prim_paths_expr=f"/World/{self.name}",
            name=self.name,
            positions=positions,
            orientations=orientations,
        )
        world.scene.add(self.object)

    def get_obj_pos(self):
        """Return the current world position and orientation of the object."""
        return {
            'pos': self.object.get_world_poses()[0].tolist()[0],
            'rot': self.object.get_world_poses()[1].tolist()[0],
        }
