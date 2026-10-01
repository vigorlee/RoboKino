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
"""Controllers for the PullPushDrawer task."""

import typing

import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators import controllers as manipulators_controllers
from isaacsim.robot.manipulators.grippers import ParallelGripper

from controllers.motion import RMPFlowController
from controllers.tasks.utils import (combine_convex, mix_sin,
                                     normalize_events_dt)


class LeftArmPickPlaceController(manipulators_controllers.PickPlaceController):
    """Left-arm pick-and-place state machine that drops an object into the drawer.

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
    - Phase 9: Retreat to the original xy position.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt=None,
        end_effector_initial_height: float = 1.2,
    ) -> None:
        """Initialize the pick-and-place state machine.

        Args:
            name: Controller name.
            gripper: Parallel gripper used to grasp the object.
            robot_articulation: Articulation of the left arm.
            events_dt: Per-phase time increments (length up to 10); ``None`` uses defaults.
            end_effector_initial_height: Height used for the approach phases.
        """
        if events_dt is None:
            events_dt = [
                0.01, 0.01, 0.5, 0.2, 0.008, 0.004, 0.008, 0.4, 0.01, 0.01
            ]

        super().__init__(
            name=name,
            cspace_controller=RMPFlowController(
                name=name + '_cspace_controller',
                robot_articulation=robot_articulation),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=end_effector_initial_height,
        )

    def _get_alpha(self):
        """Return the xy-interpolation weight for the current phase."""
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 0
        else:
            raise ValueError()


class RightArmPullPushController(BaseController):
    """Right-arm state machine that pulls then pushes a drawer.

    Each phase runs for 1 second of internal state-machine time; the ``dt`` of
    each phase is specified separately via ``events_dt``.

    - Phase 0: Move end-effector above the drawer handle at the initial height.
    - Phase 1: Lower end-effector down to the drawer-handle height.
    - Phase 2: Wait for the arm's inertia to settle.
    - Phase 3: Close the gripper to grasp the handle.
    - Phase 4: Pull the drawer open.
    - Phase 5: Wait for the arm's inertia to settle after pulling.
    - Phase 6: Push the drawer closed.
    - Phase 7: Open the gripper to release the handle.
    - Phase 8: Retreat to the pull xy position and lift back to the initial height.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        gripper: ParallelGripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Initialize the pull-push state machine.

        Args:
            name: Controller name.
            robot_articulation: Articulation of the right arm.
            gripper: Parallel gripper on the right arm.
            end_effector_initial_height: Initial end-effector height; defaults
                to ``0.9`` when ``None``.
            events_dt: Per-phase time increments (length up to 9).
        """
        if events_dt is None:
            events_dt = [
                0.008, 0.005, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008
            ]

        cspace_controller = RMPFlowController(
            name=name + '_cspace_controller',
            robot_articulation=robot_articulation)

        super().__init__(name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.9
        self._h0 = None
        self._events_dt = normalize_events_dt(events_dt, max_length=9)

        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        return

    def is_paused(self) -> bool:
        """Return whether the state machine is currently paused."""
        return self._pause

    def get_current_event(self) -> int:
        """Return the index of the currently-active phase."""
        return self._event

    def _get_alpha(self):
        """Return the xy-interpolation weight for the current phase."""
        if self._event < 4:
            return 0
        elif self._event in [4, 6]:
            return mix_sin(self._t)
        elif self._event == 8:
            return 1.0
        else:
            raise ValueError()

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        """Interpolate between the current xy and the target xy.

        During the push phase (phase 6) the start and end points are swapped so
        that motion goes from the pulled-open position back to the closed position.
        """
        alpha = self._get_alpha()
        if self._event == 6:
            start = np.array([target_x, target_y])
            end = np.array([current_x, current_y])
        else:
            start = np.array([current_x, current_y])
            end = np.array([target_x, target_y])
        return combine_convex(start, end, alpha)

    def _get_target_hs(self):
        """Return the target end-effector height for the current phase."""
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            h = combine_convex(self._h1, self._h0, mix_sin(self._t))
        elif self._event in [4, 6]:
            h = self._h0
        elif self._event == 8:
            h = combine_convex(self._h0, self._h1, mix_sin(self._t))
        else:
            raise ValueError()
        return h

    def forward(
        self,
        push_position: np.ndarray,
        pull_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Advance the state machine by one step and return the next action.

        Args:
            push_position: World-frame xyz of the closed-drawer handle position.
            pull_position: World-frame xyz of the fully-pulled handle position.
            current_joint_positions: Current joint positions of the right arm.
            end_effector_offset: Offset added to the target position.
            end_effector_orientation: Desired end-effector orientation.

        Returns:
            An :class:`ArticulationAction` for the current step.
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])

        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event in [2, 5]:
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action='close')
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action='open')
        else:
            if self._event in [0, 1]:
                self._current_target_x = push_position[0]
                self._current_target_y = push_position[1]
                self._h0 = push_position[2]

            interpolated_xy = self._get_interpolated_xy(
                pull_position[0],
                pull_position[1],
                self._current_target_x,
                self._current_target_y,
            )
            target_height = self._get_target_hs()
            position_target = np.array([
                interpolated_xy[0] + end_effector_offset[0],
                interpolated_xy[1] + end_effector_offset[1],
                target_height + end_effector_offset[2],
            ])
            orientation_target = end_effector_orientation

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=orientation_target,
            )

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joint_positions

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Reset the state machine and its underlying cspace controller.

        Args:
            end_effector_initial_height: Optional override for the approach
                height; the previous value is kept when ``None``.
            events_dt: Optional override for the per-phase time increments.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = normalize_events_dt(events_dt, max_length=9)
        return

    def is_done(self) -> bool:
        """Return whether the state machine has finished all phases."""
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pause the state machine; subsequent forwards will emit no-ops."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resume the state machine after a previous ``pause`` call."""
        self._pause = False
        return
