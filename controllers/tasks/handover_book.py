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
"""Controllers for the HandoverBook task."""

import typing

import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper

from controllers.motion import RMPFlowController
from controllers.tasks.utils import (combine_convex, interpolate_xy, mix_sin,
                                     normalize_events_dt)


class LeftArmPickPlaceController(BaseController):
    """Left-arm pick-and-place state machine that hands a book over.

    Each phase runs for 1 second of internal state-machine time; the ``dt`` of
    each phase is specified separately via ``events_dt``.

    - Phase 0: Move end-effector above the object.
    - Phase 1: Lower end-effector down to the grasping height.
    - Phase 2: Wait for the arm's inertia to settle.
    - Phase 3: Close the gripper.
    - Phase 4: Lift the object upward while slerping the end-effector
      orientation from the picking orientation to the handover orientation.
    - Phase 5: Move end-effector toward the handover position.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        gripper: ParallelGripper,
        events_dt: typing.Optional[typing.List[float]] = None,
        end_effector_initial_height: typing.Optional[float] = None,
    ) -> None:
        """Initialize the left-arm pick-and-place state machine.

        Args:
            name: Controller name.
            robot_articulation: Articulation of the left arm.
            gripper: Parallel gripper on the left arm.
            events_dt: Per-phase time increments (length up to 6).
            end_effector_initial_height: Initial end-effector height; defaults
                to ``1.1`` when ``None``.
        """
        if events_dt is None:
            events_dt = [0.01, 0.01, 1, 0.03, 0.08, 0.01]

        cspace_controller = RMPFlowController(
            name=name + '_cspace_controller',
            robot_articulation=robot_articulation)

        super().__init__(name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 1.1
        self._h0 = None
        self._events_dt = normalize_events_dt(events_dt, max_length=6)

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
        if self._event < 5:
            return 0
        elif self._event == 5:
            return mix_sin(self._t)
        else:
            raise ValueError()

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        """Interpolate between the current xy and the target xy."""
        return interpolate_xy(target_x, target_y, current_x, current_y,
                              self._get_alpha())

    def _get_target_hs(self, target_height):
        """Return the target end-effector height for the current phase."""
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            h = combine_convex(self._h1, self._h0, mix_sin(self._t))
        elif self._event == 4:
            h = combine_convex(self._h0, target_height + 0.02,
                               mix_sin(self._t))
        elif self._event == 5:
            h = target_height + 0.02
        else:
            raise ValueError()
        return h

    def _get_interpolated_orientation(self, picking_orientation,
                                      placing_orientation):
        """Slerp between the picking and placing orientations."""
        if self._event <= 3:
            return picking_orientation
        elif self._event == 4:
            alpha = mix_sin(self._t)
            dot = np.dot(picking_orientation, placing_orientation)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot) * alpha
            rot_a = picking_orientation * np.cos(theta)
            rot_b = ((placing_orientation - picking_orientation * dot) /
                     np.sqrt(1 - dot * dot + 1e-8) * np.sin(theta))
            interpolated = rot_a + rot_b
            return interpolated / (np.linalg.norm(interpolated) + 1e-8)
        elif self._event == 5:
            return placing_orientation
        else:
            raise ValueError()

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        picking_orientation: typing.Optional[np.ndarray] = None,
        placing_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Advance the state machine by one step and return the next action.

        Args:
            picking_position: World-frame xyz of the book grasping pose.
            placing_position: World-frame xyz of the handover target pose.
            current_joint_positions: Current joint positions of the left arm.
            end_effector_offset: Offset added to the target position.
            picking_orientation: Desired end-effector orientation during picking.
            placing_orientation: Desired end-effector orientation during handover.

        Returns:
            An :class:`ArticulationAction` for the current step.
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])

        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event == 2:
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action='close')
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]

            interpolated_xy = self._get_interpolated_xy(
                placing_position[0],
                placing_position[1],
                self._current_target_x,
                self._current_target_y,
            )
            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array([
                interpolated_xy[0] + end_effector_offset[0],
                interpolated_xy[1] + end_effector_offset[1],
                target_height + end_effector_offset[2],
            ])
            orientation_target = self._get_interpolated_orientation(
                picking_orientation, placing_orientation)

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
            self._events_dt = normalize_events_dt(events_dt, max_length=6)
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


class RightArmPickPlaceController(BaseController):
    """Right-arm state machine that receives the book and places it on the bookend.

    Each phase runs for 1 second of internal state-machine time; the ``dt`` of
    each phase is specified separately via ``events_dt``.

    - Phase 0: Move end-effector above the handover position at the initial height.
    - Phase 1: Lower end-effector down to the grasping height.
    - Phase 2: Wait for the arm's inertia to settle.
    - Phase 3: Close the gripper.
    - Phase 4: Hold position and wait for stabilization.
    - Phase 5: Lift the book vertically to a safe height above the placing position.
    - Phase 6: Move horizontally toward the placing position while switching
      to the placing orientation.
    - Phase 7: Lower the book to the placing height.
    - Phase 8: Open the gripper to release the book.
    - Phase 9: Move end-effector back above the placing position.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        gripper: ParallelGripper,
        events_dt: typing.Optional[typing.List[float]] = None,
        end_effector_initial_height: typing.Optional[float] = None,
    ) -> None:
        """Initialize the right-arm pick-and-place state machine.

        Args:
            name: Controller name.
            robot_articulation: Articulation of the right arm.
            gripper: Parallel gripper on the right arm.
            events_dt: Per-phase time increments (length up to 10).
            end_effector_initial_height: Initial end-effector height; defaults
                to ``1.1`` when ``None``.
        """
        if events_dt is None:
            events_dt = [
                0.008, 0.008, 1, 0.03, 0.01, 0.01, 0.01, 0.01, 0.1, 0.008
            ]

        cspace_controller = RMPFlowController(
            name=name + '_cspace_controller',
            robot_articulation=robot_articulation)

        super().__init__(name=name)
        self._event = 0
        self._t = 0
        self._h1 = (
            end_effector_initial_height
            if end_effector_initial_height is not None else 1.1)
        self._h0 = None
        self._events_dt = normalize_events_dt(events_dt, max_length=10)

        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False

    def is_paused(self) -> bool:
        """Return whether the state machine is currently paused."""
        return self._pause

    def get_current_event(self) -> int:
        """Return the index of the currently-active phase."""
        return self._event

    def _get_alpha(self):
        """Return the xy-interpolation weight for the current phase."""
        if self._event <= 5:
            return 0
        elif self._event == 6:
            return mix_sin(self._t)
        elif self._event in [7, 9]:
            return 1.0
        else:
            raise ValueError()

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        """Interpolate between the current xy and the target xy."""
        return interpolate_xy(target_x, target_y, current_x, current_y,
                              self._get_alpha())

    def _get_target_hs(self, target_height):
        """Return the target end-effector height for the current phase."""
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            h = combine_convex(self._h1, self._h0, mix_sin(self._t))
        elif self._event == 5:
            h = combine_convex(self._h0, target_height + 0.2, mix_sin(self._t))
        elif self._event == 6:
            h = target_height + 0.2
        elif self._event == 7:
            h = combine_convex(target_height + 0.2, target_height - 0.03,
                               mix_sin(self._t))
        elif self._event == 9:
            h = combine_convex(target_height - 0.03, self._h1,
                               mix_sin(self._t))
        else:
            raise ValueError()
        return h

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        picking_orientation: typing.Optional[np.ndarray] = None,
        placing_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Advance the state machine by one step and return the next action.

        Args:
            picking_position: World-frame xyz of the handover pose where the
                right arm receives the book.
            placing_position: World-frame xyz of the placing pose on the bookend.
            current_joint_positions: Current joint positions of the right arm.
            end_effector_offset: Offset added to the target position.
            picking_orientation: Desired end-effector orientation during picking.
            placing_orientation: Desired end-effector orientation during placing.

        Returns:
            An :class:`ArticulationAction` for the current step.
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event in [2, 4]:
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action='close')
        elif self._event == 8:
            target_joint_positions = self._gripper.forward(action='open')
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]

            interpolated_xy = self._get_interpolated_xy(
                placing_position[0],
                placing_position[1],
                self._current_target_x,
                self._current_target_y,
            )
            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array([
                interpolated_xy[0] + end_effector_offset[0],
                interpolated_xy[1] + end_effector_offset[1],
                target_height + end_effector_offset[2],
            ])
            orientation_target = (
                picking_orientation
                if self._event <= 5 else placing_orientation)

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
            self._events_dt = normalize_events_dt(events_dt, max_length=10)

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
