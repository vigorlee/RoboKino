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
"""Shared utilities for task-specific scripted controllers."""

from typing import List, Union

import numpy as np


def mix_sin(t: float) -> float:
    """Smooth-step interpolation weight in ``[0, 1]`` for ``t`` in ``[0, 1]``.

    Args:
        t: Normalized phase progress in ``[0, 1]``.

    Returns:
        ``0.5 * (1 - cos(t * pi))``, a smooth-step weight whose derivative
        vanishes at the endpoints.
    """
    return 0.5 * (1 - np.cos(t * np.pi))


def combine_convex(a, b, alpha: float):
    """Return the convex combination ``(1 - alpha) * a + alpha * b``.

    Args:
        a: Value at ``alpha = 0`` (scalar or ``np.ndarray``).
        b: Value at ``alpha = 1`` (scalar or ``np.ndarray``).
        alpha: Interpolation weight, typically in ``[0, 1]``.

    Returns:
        ``(1 - alpha) * a + alpha * b``, with the same shape/type as the
        broadcast result of ``a`` and ``b``.
    """
    return (1 - alpha) * a + alpha * b


def interpolate_xy(
    target_x: float,
    target_y: float,
    current_x: float,
    current_y: float,
    alpha: float,
) -> np.ndarray:
    """Interpolate between the current xy and the target xy.

    Args:
        target_x: Target x-coordinate (reached when ``alpha = 1``).
        target_y: Target y-coordinate (reached when ``alpha = 1``).
        current_x: Current x-coordinate (returned when ``alpha = 0``).
        current_y: Current y-coordinate (returned when ``alpha = 0``).
        alpha: Interpolation weight in ``[0, 1]``.

    Returns:
        The interpolated xy as an ``np.ndarray`` of shape ``(2,)``.
    """
    return combine_convex(
        np.array([current_x, current_y]),
        np.array([target_x, target_y]),
        alpha,
    )


def normalize_events_dt(
    events_dt: Union[List[float], np.ndarray],
    max_length: int,
) -> List[float]:
    """Validate and normalize a per-phase ``events_dt`` schedule.

    Args:
        events_dt: Per-phase time increments as a list or numpy array.
        max_length: Maximum allowed length of ``events_dt``.

    Returns:
        ``events_dt`` as a Python list.

    Raises:
        Exception: If ``events_dt`` is neither a list nor a numpy array, or
            if its length exceeds ``max_length``.
    """
    if not isinstance(events_dt, np.ndarray) and not isinstance(
            events_dt, list):
        raise Exception('events dt need to be list or numpy array')
    if isinstance(events_dt, np.ndarray):
        events_dt = events_dt.tolist()
    if len(events_dt) > max_length:
        raise Exception(
            f'events dt length must be less than or equal to {max_length}')
    return events_dt
