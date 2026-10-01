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
"""Shared utilities used across task modules."""

from typing import Sequence

import numpy as np
from isaacsim.core.prims import XFormPrim
from scipy.spatial.transform import Rotation as R

# Default offset applied from the gripper base to the grasping point, in metres.
DEFAULT_EE_OFFSET = (0.0, 0.0, 0.13)
# Default maximum end-effector-to-object distance that still counts as grasped.
DEFAULT_GRIPPER_THRESHOLD = 0.1


def _get_prim_world_pose(prim_path: str):
    """Return the world pose ``(position, quaternion)`` of the given prim."""
    prim_view = XFormPrim(
        prim_paths_expr=prim_path,
        name=f"tmp_{prim_path.split('/')[-1]}",
    )
    pos, quat = prim_view.get_world_poses()
    return np.array(pos[0]), np.array(quat[0])


def get_prim_world_pos(world, prim_path: str):
    """Return the world-frame position ``(x, y, z)`` of the given prim."""
    pos, _ = _get_prim_world_pose(prim_path)
    return pos


def get_prim_world_quat(world, prim_path: str):
    """Return the world-frame quaternion ``[w, x, y, z]`` of the given prim."""
    _, quat = _get_prim_world_pose(prim_path)
    return quat


def generate_point_in_circle(origin_x, origin_y, origin_z, radius=0.02):
    """Sample a random point uniformly inside a circle on the XY plane.

    Args:
        origin_x: X-coordinate of the circle centre.
        origin_y: Y-coordinate of the circle centre.
        origin_z: Z-coordinate applied unchanged to the sampled point.
        radius: Radius of the circle in world units.

    Returns:
        A ``[x, y, z]`` list describing the sampled point.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    r = radius * np.sqrt(np.random.uniform(0, 1))
    x = origin_x + r * np.cos(theta)
    y = origin_y + r * np.sin(theta)
    return [x, y, origin_z]


def check_gripper(
    env,
    object_name: str,
    hand: str = 'right',
    threshold: float = DEFAULT_GRIPPER_THRESHOLD,
    ee_offset: Sequence[float] = DEFAULT_EE_OFFSET,
) -> bool:
    """Check whether the specified gripper is close enough to the target object.

    Args:
        env: The environment exposing ``robot`` and ``world`` attributes.
        object_name: Name of the scene object to check against.
        hand: Either ``"left"`` or ``"right"``.
        threshold: Maximum end-effector-to-object distance for a valid grasp.
        ee_offset: Offset, expressed in the end-effector's local frame, that is
            rotated into the world frame before the distance measurement.

    Returns:
        ``True`` if the distance from the (offset) end-effector to the object
        position is at most ``threshold``.
    """
    manipulator = (
        env.robot.left_manipulator
        if hand == 'left' else env.robot.right_manipulator)
    # Isaac Sim returns the EE pose as (position, quaternion) where the
    # quaternion uses the [w, x, y, z] convention. ``ee_offset`` is defined in
    # the end-effector's local frame, so it must be rotated by the EE's world
    # orientation before being added to the EE's world position.
    ee_pos, ee_quat = manipulator.end_effector.get_world_pose()
    ee_pos = np.array(ee_pos)
    ee_quat = np.array(ee_quat)
    ee_rot = R.from_quat(ee_quat[[1, 2, 3, 0]])
    current_ee_pos = ee_pos + ee_rot.apply(np.array(ee_offset))
    object_pos = np.array(
        env.world.scene.get_object(object_name).get_world_poses()[0][0])
    return float(np.linalg.norm(current_ee_pos - object_pos)) <= threshold
