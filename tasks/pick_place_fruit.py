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
"""PickPlaceFruit task: pick a fruit off the table and place it on a plate."""

import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import check_gripper, generate_point_in_circle  # noqa: E402


class PickPlaceFruit:
    """Scene definition for the PickPlaceFruit task.

    Holds the scene objects (fruit, plate, table) and a :class:`Checker`
    instance that decides when the task is complete.
    """

    # Randomization radius applied to the fruit and plate at each reset.
    RANDOMIZATION_RADIUS = 0.03
    # Additional noise amplitude (reserved for future use).
    RANDOMIZATION_RADIUS_NOISE = 0.06

    def __init__(self):
        """Create the task scene and its success checker."""
        self.fruit_name = os.environ.get('FRUIT_NAME', 'banana')
        self.plate_name = os.environ.get('PLATE_NAME', 'plate')
        self.table_name = os.environ.get('TABLE_NAME', 'kitchen_table')

        self.objects = [
            BaseObject(self.fruit_name),
            BaseObject(self.plate_name)
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(self.fruit_name, self.plate_name)

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.fruit_name or obj.name == self.plate_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))

            # Apply the new default state before moving the object.
            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.fruit_name or obj.name == self.plate_name:
                print(
                    'obj.name:',
                    obj.name,
                    'new position:',
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    """Success checker for the PickPlaceFruit task."""

    # Maximum xy distance between the fruit and the plate for a successful place.
    PLACE_THRESHOLD_MIN = 0.06
    # Maximum end-effector-to-object distance that still counts as grasped.
    GRIPPER_THRESHOLD = 0.1

    def __init__(self, fruit_name, plate_name):
        """Initialize the checker with the scene object names."""
        self.fruit_name = fruit_name
        self.plate_name = plate_name

    def check_gripper(self, env):
        """Check whether the right gripper is close enough to grasp the fruit."""
        return check_gripper(
            env,
            object_name=self.fruit_name,
            hand='right',
            threshold=self.GRIPPER_THRESHOLD,
        )

    def check(self, env):
        """Return ``True`` once the fruit is on the plate and the gripper is released."""
        fruit = env.world.scene.get_object(self.fruit_name)
        plate = env.world.scene.get_object(self.plate_name)
        if fruit is None or plate is None:
            raise ValueError('Fruit or plate object does not exist.')

        fruit_pos = fruit.get_world_poses()[0][0]
        plate_pos = plate.get_world_poses()[0][0]

        distance_xy = np.linalg.norm(fruit_pos[:2] - plate_pos[:2])

        # Task succeeds when the fruit is on the plate and no longer grasped.
        return (distance_xy <= self.PLACE_THRESHOLD_MIN
                and not self.check_gripper(env))
