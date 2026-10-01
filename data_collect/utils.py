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
"""Shared utilities for data collection scripts."""

import os
from typing import Any, Dict

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


def relative_to_world_position(env,
                               relative_position: np.ndarray,
                               base_name: str = None) -> np.ndarray:
    """Convert a position expressed in the robot base frame to world frame.

    Args:
        env: Isaac Sim environment instance.
        relative_position: Position in the base frame, ``[x, y, z]``.
        base_name: Robot base name. If ``None``, it is read from the
            ``BASE_NAME`` environment variable.

    Returns:
        The corresponding world-frame position ``[x, y, z]``.
    """
    if base_name is None:
        base_name = os.environ.get('BASE_NAME', 'base_kitchen')

    base_position = np.array([0.0, 0.0, 0.0])
    base_orientation = np.array([1.0, 0.0, 0.0, 0.0])

    if hasattr(env.robot, 'robot_cfg') and base_name in env.robot.robot_cfg:
        base_cfg = env.robot.robot_cfg[base_name]
        base_position = np.array(base_cfg.get('position', [0.0, 0.0, 0.0]))
        base_orientation = np.array(
            base_cfg.get('orientation', [1.0, 0.0, 0.0, 0.0]))

    # base_orientation is stored as [w, x, y, z]; convert to scipy's
    # [x, y, z, w] ordering before applying the rotation.
    base_rot = R.from_quat(base_orientation[[1, 2, 3, 0]])
    world_position = base_position + base_rot.apply(relative_position)

    return world_position


def init_hdf5(data_path: str, data: Dict[str, Any]) -> None:
    """Initialize an HDF5 file with empty, extensible datasets.

    Args:
        data_path: Target HDF5 file path.
        data: Initial sample dict containing camera images and ``qpos``.
    """
    with h5py.File(data_path, 'w') as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')

        dof_len = len(data['qpos'])

        for name, array in data.items():
            if 'cam' in name:
                array = np.asarray(array)
                assert array.ndim == 3 and array.shape[-1] in (3, 4)
                h, w, c = array.shape
                _ = image.create_dataset(
                    name,
                    shape=(0, h, w, c),
                    maxshape=(None, h, w, c),
                    dtype='uint8',
                    chunks=(1, h, w, c),
                )

        _ = obs.create_dataset(
            'qpos',
            shape=(0, dof_len),
            maxshape=(None, dof_len),
            dtype=float,
            chunks=True,
        )
        _ = root.create_dataset(
            'action',
            shape=(0, dof_len),
            maxshape=(None, dof_len),
            dtype=float,
            chunks=True,
        )


def append_hdf5(data_path: str, data: Dict[str, Any]) -> None:
    """Append one step of data to an existing HDF5 file.

    Args:
        data_path: Target HDF5 file path.
        data: Sample dict containing camera images, ``qpos`` and ``action``.
    """
    with h5py.File(data_path, 'a') as root:
        images = root['observations/images']
        obs = root['observations']

        for name, value in data.items():
            if 'cam' in name and isinstance(value, np.ndarray):
                images[name].resize(images[name].shape[0] + 1, axis=0)
                images[name][-1, ...] = value
            elif 'qpos' in name:
                obs[name].resize(obs[name].shape[0] + 1, axis=0)
                obs[name][-1, ...] = value
            elif 'action' in name:
                root[name].resize(root[name].shape[0] + 1, axis=0)
                root[name][-1, ...] = value


def extract_state_from_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the state (camera images and joint positions) from an observation.

    Args:
        obs: Raw observation dict.

    Returns:
        A dict with camera images and ``qpos``.
    """
    state = {name: value for name, value in obs.items() if 'cam' in name}
    if 'piper' in obs and 'dof_pos' in obs['piper']:
        q_pos = np.array(list(obs['piper']['dof_pos'].values()), dtype=float)
        state['qpos'] = q_pos
    return state


def extract_action_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    """Extract the action (joint positions) from an observation.

    Args:
        obs: Raw observation dict.

    Returns:
        Joint positions as a 1-D ``np.ndarray``.
    """
    if 'piper' in obs and 'dof_pos' in obs['piper']:
        return np.array(list(obs['piper']['dof_pos'].values()), dtype=float)
    return np.array([])
