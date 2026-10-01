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
"""HandoverBook task: hand a book from one arm to the other and place it on the bookend."""

import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import check_gripper, generate_point_in_circle  # noqa: E402


class HandoverBook:
    """Scene definition for the HandoverBook task.

    Holds the scene objects (book, bookend, table) and a :class:`Checker`
    instance that decides when the task is complete.
    """

    # Randomization radii applied to object positions at each reset.
    BOOK_RANDOMIZATION_RADIUS = 0.02
    BOOKEND_RANDOMIZATION_RADIUS = 0.01
    # Additional noise amplitude (reserved for future use).
    BOOK_RANDOMIZATION_RADIUS_NOISE = 0.04
    BOOKEND_RANDOMIZATION_RADIUS_NOISE = 0.02

    def __init__(self):
        """Create the task scene and its success checker."""
        self.book_name = os.environ.get('BOOK_NAME', 'book')
        self.bookend_name = os.environ.get('BOOKEND_NAME', 'bookend')
        self.table_name = os.environ.get('TABLE_NAME', 'kitchen_table')

        self.objects = [
            BaseObject(self.book_name),
            BaseObject(self.bookend_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(
            book_name=self.book_name, bookend_name=self.bookend_name)

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.book_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.BOOK_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))
            elif obj.name == self.bookend_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.BOOKEND_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.book_name or obj.name == self.bookend_name:
                print(
                    'obj.name:',
                    obj.name,
                    'new position:',
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    """Success checker for the HandoverBook task."""

    # Minimum xy distance between the book and the bookend at placement.
    PLACE_THRESHOLD_MIN = 0.15
    # Maximum xy distance between the book and the bookend at placement.
    PLACE_THRESHOLD_MAX = 0.2
    # Valid placement height range (world-frame z).
    HEIGHT_MIN = 0.8
    HEIGHT_MAX = 0.88
    # Maximum end-effector-to-object distance that still counts as grasped.
    GRIPPER_THRESHOLD = 0.2

    def __init__(self, book_name, bookend_name):
        """Initialize the checker with the scene object names."""
        self.book_name = book_name
        self.bookend_name = bookend_name

    def check_gripper(self, env, hand='left'):
        """Check whether the specified gripper is close enough to grasp the book."""
        return check_gripper(
            env,
            object_name=self.book_name,
            hand=hand,
            threshold=self.GRIPPER_THRESHOLD,
        )

    def check(self, env):
        """Return ``True`` if the book is placed at the correct height and distance."""
        book = env.world.scene.get_object(self.book_name)
        bookend = env.world.scene.get_object(self.bookend_name)
        if book is None or bookend is None:
            raise ValueError('Book or bookend object does not exist.')

        book_pos = book.get_world_poses()[0][0]
        book_height = book_pos[2]

        bookend_pos = bookend.get_world_poses()[0][0]

        distance_xy = np.linalg.norm(book_pos[:2] - bookend_pos[:2])

        # A valid placement must satisfy both the height and the distance check.
        height_check = self.HEIGHT_MIN <= book_height <= self.HEIGHT_MAX
        distance_check = (
            self.PLACE_THRESHOLD_MIN <= distance_xy <=
            self.PLACE_THRESHOLD_MAX)

        return height_check and distance_check
