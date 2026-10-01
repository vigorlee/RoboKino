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
"""ScrewPitcherLid task: place the pitcher lid on the pitcher in the target orientation."""

import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import check_gripper  # noqa: E402
from tasks.utils import generate_point_in_circle  # noqa: E402
from tasks.utils import get_prim_world_pos  # noqa: E402


class ScrewPitcherLid:
    """Scene definition for the ScrewPitcherLid task.

    Holds the scene objects (pitcher, lid, table) and a :class:`Checker`
    instance that decides when the task is complete.
    """

    # Randomization radii applied at each reset.
    PITCHER_RANDOMIZATION_RADIUS = 0.005
    LID_RANDOMIZATION_RADIUS = 0.02
    # Additional noise amplitude (reserved for future use).
    PITCHER_RANDOMIZATION_RADIUS_NOISE = 0.01
    LID_RANDOMIZATION_RADIUS_NOISE = 0.04

    def __init__(self):
        """Create the task scene and its success checker."""
        self.pitcher_name = os.environ.get('PITCHER_NAME', 'pitcher')
        self.lid_name = os.environ.get('LID_NAME', 'pitcherlid')
        self.table_name = os.environ.get('TABLE_NAME', 'kitchen_table')

        self.objects = [
            BaseObject(self.pitcher_name),
            BaseObject(self.lid_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(
            lid_name=self.lid_name, pitcher_name=self.pitcher_name)

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.pitcher_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.PITCHER_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))
            elif obj.name == self.lid_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.LID_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.pitcher_name or obj.name == self.lid_name:
                print(
                    'obj.name:',
                    obj.name,
                    'new position:',
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    """Success checker for the ScrewPitcherLid task."""

    # Maximum xy distance between the lid and the pitcher opening.
    PLACE_THRESHOLD = 0.015
    # Maximum orientation difference (degrees) between the lid and the target pose.
    ORIENTATION_THRESHOLD_DEG = 5.0
    # Target lid orientation quaternion ``[w, x, y, z]``.
    TARGET_LID_QUAT = np.array([0.77, 0.00303, -0.00313, 0.64])
    # End-effector offsets used by gripper checks (kept for reference).
    RIGHT_EE_OFFSET = np.array([-0.01, 0.0, -0.17])
    LEFT_EE_OFFSET = np.array([0.413, 0.03, -0.14])
    # Maximum end-effector-to-object distance that still counts as grasped.
    GRIPPER_THRESHOLD = 0.1

    def __init__(self, lid_name, pitcher_name):
        """Initialize the checker with the scene object names."""
        self.lid_name = lid_name
        self.pitcher_name = pitcher_name

    def check_gripper(self, env):
        """Check whether the right gripper is close enough to grasp the lid."""
        return check_gripper(
            env,
            object_name=self.lid_name,
            hand='right',
            threshold=self.GRIPPER_THRESHOLD,
        )

    def check(self, env):
        """Return ``True`` once the lid sits on the pitcher with the desired pose."""
        lid = env.world.scene.get_object(self.lid_name)
        pitcher = env.world.scene.get_object(self.pitcher_name)
        if lid is None or pitcher is None:
            raise ValueError('Lid or pitcher object does not exist.')

        lid_pos_raw, lid_orient_raw = lid.get_world_poses()
        lid_pos = np.array(lid_pos_raw[0])
        lid_orient = np.array(lid_orient_raw[0])
        pitcher_pos = get_prim_world_pos(
            env.world, f"/World/{self.pitcher_name}/pitcher_base/Visuals")

        distance_xy = np.linalg.norm(lid_pos[:2] - pitcher_pos[:2])

        if np.linalg.norm(lid_orient) > 0:
            lid_q = lid_orient / np.linalg.norm(lid_orient)
            target_q = self.TARGET_LID_QUAT / np.linalg.norm(
                self.TARGET_LID_QUAT)
            # Use the absolute dot product so that q and -q are treated as the
            # same rotation.
            dot = float(np.clip(np.abs(np.dot(lid_q, target_q)), -1.0, 1.0))
            angle_diff_deg = np.degrees(2.0 * np.arccos(dot))
        else:
            angle_diff_deg = 180.0

        return (distance_xy <= self.PLACE_THRESHOLD
                and angle_diff_deg < self.ORIENTATION_THRESHOLD_DEG)
