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
"""CloseBox task: place a nut into a box and close the box lid."""

import os

import numpy as np
import rootutils
from scipy.spatial.transform import Rotation as R

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import check_gripper  # noqa: E402
from tasks.utils import generate_point_in_circle  # noqa: E402
from tasks.utils import get_prim_world_quat  # noqa: E402


class CloseBox:
    """Scene definition for the CloseBox task.

    Holds the scene objects (nut, box, table) and a :class:`Checker` instance
    that decides when the task is complete.
    """

    # Randomization radius applied to the nut and box at each reset.
    RANDOMIZATION_RADIUS = 0.03
    # Additional noise amplitude (reserved for future use).
    RANDOMIZATION_RADIUS_NOISE = 0.06

    def __init__(self):
        """Create the task scene and its success checker."""
        self.nut_name = os.environ.get('NUT_NAME', 'nut')
        self.box_name = os.environ.get('BOX_NAME', 'box')
        self.table_name = os.environ.get('TABLE_NAME', 'industry_table')

        self.objects = [
            BaseObject(self.nut_name),
            BaseObject(self.box_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(nut_name=self.nut_name, box_name=self.box_name)

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.nut_name or obj.name == self.box_name:
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
            if obj.name == self.nut_name or obj.name == self.box_name:
                print(
                    'obj.name:',
                    obj.name,
                    'new position:',
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    """Success checker for the CloseBox task."""

    # Maximum xy distance from the nut to the box centre that still counts
    # as "inside the box".
    PLACE_THRESHOLD = 0.15
    # Angular range (degrees) on the pitch axis for which the lid counts as closed.
    CLOSE_THRESHOLD_MIN_DEG = 10.0
    CLOSE_THRESHOLD_MAX_DEG = 15.0
    # Maximum end-effector-to-object distance that still counts as grasped.
    GRIPPER_THRESHOLD = 0.1

    def __init__(self, nut_name, box_name):
        """Initialize the checker with the scene object names."""
        self.nut_name = nut_name
        self.box_name = box_name

    def check_gripper(self, env):
        """Check whether the left gripper is close enough to grasp the nut."""
        return check_gripper(
            env,
            object_name=self.nut_name,
            hand='left',
            threshold=self.GRIPPER_THRESHOLD,
        )

    def check(self, env):
        """Return ``True`` if the nut is inside the box and the lid is closed."""
        nut = env.world.scene.get_object(self.nut_name)
        box = env.world.scene.get_object(self.box_name)
        if nut is None or box is None:
            raise ValueError('Nut or box object does not exist.')

        nut_pos = np.array(nut.get_world_poses()[0][0])
        box_pos = np.array(box.get_world_poses()[0][0])
        box_lid_quat = get_prim_world_quat(env.world,
                                           f"/World/{self.box_name}/box_lid")
        box_lid_euler = R.from_quat([
            box_lid_quat[1], box_lid_quat[2], box_lid_quat[3], box_lid_quat[0]
        ]).as_euler(
            'xyz',
            degrees=True,
        )

        distance_xy = np.linalg.norm(nut_pos[:2] - box_pos[:2])

        return (distance_xy <= self.PLACE_THRESHOLD
                and self.CLOSE_THRESHOLD_MIN_DEG <= box_lid_euler[1] <=
                self.CLOSE_THRESHOLD_MAX_DEG)
