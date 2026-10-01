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
"""Controller for the PickPlaceFruit task."""

import typing

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators import controllers as manipulators_controllers
from isaacsim.robot.manipulators.grippers import ParallelGripper

from controllers.motion import RMPFlowController


class PickPlaceFruitController(manipulators_controllers.PickPlaceController):
    """Pick-and-place controller without the final retreat-to-old-xy phase.

    The state machine progresses through the following phases:

    - Phase 0: Move end-effector above the picking position at the initial height.
    - Phase 1: Lower end-effector down to the grasping height.
    - Phase 2: Wait for the arm's inertia to settle.
    - Phase 3: Close the gripper.
    - Phase 4: Lift the object upward, keeping the gripper closed.
    - Phase 5: Move horizontally toward the placing xy position.
    - Phase 6: Lower the end-effector toward the placing height.
    - Phase 7: Open the gripper to release the object.
    - Phase 8: Lift the end-effector back to the initial height.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt=None,
        end_effector_initial_height: float = 0.9,
    ) -> None:
        """Initialize the fruit pick-and-place state machine.

        Args:
            name: Controller name.
            gripper: Parallel gripper used to grasp the fruit.
            robot_articulation: Articulation of the manipulating arm.
            events_dt: Per-phase time increments (must have length 9).
            end_effector_initial_height: Height used for the approach phases.
        """
        if events_dt is None:
            events_dt = [
                0.008, 0.008, 0.5, 0.2, 0.008, 0.004, 0.008, 0.4, 0.008
            ]

        if len(events_dt) != 9:
            raise Exception('events dt length must be 9')

        super().__init__(
            name=name,
            cspace_controller=RMPFlowController(
                name=name + '_cspace_controller',
                robot_articulation=robot_articulation),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=end_effector_initial_height,
        )

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Reset the state machine.

        Args:
            end_effector_initial_height: Optional override for the approach height.
            events_dt: Optional override for the per-phase time increments
                (must have length 9 when provided).
        """
        if events_dt is not None:
            if not isinstance(events_dt, np.ndarray) and not isinstance(
                    events_dt, list):
                raise Exception('events dt need to be list or numpy array')
            elif isinstance(events_dt, np.ndarray):
                events_dt = events_dt.tolist()
            if len(events_dt) != 9:
                raise Exception('events dt length must be 9')

        super().reset(
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )
