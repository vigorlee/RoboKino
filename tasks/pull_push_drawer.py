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
"""PullPushDrawer task: open a drawer, place an object inside, and close it."""

import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import check_gripper  # noqa: E402
from tasks.utils import generate_point_in_circle  # noqa: E402
from tasks.utils import get_prim_world_pos  # noqa: E402


class PullPushDrawer:
    """Scene definition for the PullPushDrawer task.

    Holds the scene objects (manipulated object, drawer, table) and a
    :class:`Checker` instance that decides when the task is complete.
    """

    # Randomization radius applied to object positions at each reset.
    RANDOMIZATION_RADIUS = 0.02
    # Additional noise amplitude (reserved for future use).
    RANDOMIZATION_RADIUS_NOISE = 0.04

    def __init__(self):
        """Create the task scene and its success checker."""
        self.object_name = os.environ.get('OBJECT_NAME', 'apple_drawer')
        self.drawer_name = os.environ.get('DRAWER_NAME', 'drawer')
        self.table_name = os.environ.get('TABLE_NAME', 'kitchen_table')

        self.objects = [
            BaseObject(self.object_name),
            BaseObject(self.drawer_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(
            object_name=self.object_name, drawer_name=self.drawer_name)

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.object_name or obj.name == self.drawer_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.object_name or obj.name == self.drawer_name:
                print(
                    'obj.name:',
                    obj.name,
                    'new position:',
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    """Success checker for the PullPushDrawer task."""

    # Maximum xy distance between the object and the target position in the drawer.
    POSITION_THRESHOLD = 0.05
    # Minimum distance the drawer must be pulled open during the episode.
    MIN_OPEN_DIST = 0.10
    # Maximum distance that still counts as "drawer closed".
    MAX_CLOSED_DIST = 0.02
    # Maximum end-effector-to-object distance that still counts as grasped.
    GRIPPER_THRESHOLD = 0.1
    # Baseline offset (in metres) subtracted when measuring the drawer opening.
    DRAWER_OFFSET = 0.04646722

    def __init__(self, object_name, drawer_name):
        """Initialize the checker with the scene object names."""
        self.object_name = object_name
        self.drawer_name = drawer_name
        self.max_drawer_dist = 0.0

    def check_gripper(self, env):
        """Check whether the left gripper is close enough to grasp the object."""
        return check_gripper(
            env,
            object_name=self.object_name,
            hand='left',
            threshold=self.GRIPPER_THRESHOLD,
        )

    def _get_drawer_distance(self, env):
        """Return the current drawer opening distance in metres."""
        drawer_front_pos = get_prim_world_pos(env.world,
                                              '/World/drawer/E_drawer2_7')
        target_part_pos = get_prim_world_pos(
            env.world, '/World/drawer/E_body_1/E_part4_5')
        return np.linalg.norm(drawer_front_pos -
                              target_part_pos) - self.DRAWER_OFFSET

    def calculate_open_distance(self, env):
        """Update the running maximum of the drawer opening distance."""
        distance = self._get_drawer_distance(env)
        if distance > self.max_drawer_dist:
            self.max_drawer_dist = distance

    def check_drawer_open(self):
        """Return ``True`` if the drawer has been pulled open at least ``MIN_OPEN_DIST``."""
        return self.max_drawer_dist > self.MIN_OPEN_DIST

    def check_drawer_closed(self, env):
        """Return ``True`` if the drawer is currently closed."""
        return self._get_drawer_distance(env) < self.MAX_CLOSED_DIST

    def check(self, env):
        """Return ``True`` once the object is in place and the drawer is closed."""
        object = env.world.scene.get_object(self.object_name)
        drawer = env.world.scene.get_object(self.drawer_name)
        if object is None or drawer is None:
            raise ValueError('Object or drawer does not exist.')

        object_pos = object.get_world_poses()[0][0]
        drawer_pos = get_prim_world_pos(
            env.world, '/World/drawer/E_drawer2_7') + np.array(
                [-0.063, 0.083, 0.0018])

        distance_xy = np.linalg.norm(object_pos[:2] - drawer_pos[:2])

        return distance_xy <= self.POSITION_THRESHOLD and self.check_drawer_closed(
            env)
