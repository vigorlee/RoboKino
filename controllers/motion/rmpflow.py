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
"""RMPflow-based motion policy controller for a single 7-DoF arm."""

import os

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation


class RMPFlowController(mg.MotionPolicyController):
    """Cartesian-space controller using the RMPflow motion policy.

    The controller loads the robot description, RMPflow configuration and URDF
    files bundled under ``controllers/motion/rmpflow`` and targets the
    ``gripper_base`` frame of the arm.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 30.0,
    ) -> None:
        """Construct the RMPflow controller for a given articulation.

        Args:
            name: Controller name used for logging and bookkeeping.
            robot_articulation: The arm articulation to be controlled.
            physics_dt: Physics time step used by the motion policy.
        """
        motion_path = os.path.dirname(__file__)
        cfg_path = os.path.join(motion_path, 'rmpflow')
        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(cfg_path,
                                                'piper_description.yaml'),
            rmpflow_config_path=os.path.join(cfg_path,
                                             'piper_rmpflow_config.yaml'),
            urdf_path=os.path.join(motion_path, 'piper_description.urdf'),
            end_effector_frame_name='gripper_base',
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose(
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    def reset(self):
        """Reset the motion policy and restore the cached robot base pose."""
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
