"""
Unit tests for trajectory_tracker.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocketpy.gnc.guidance.trajectory import (
    ReferenceTrajectory,
    Waypoint,
)
from rocketpy.gnc.guidance.trajectory_tracker import (
    TrajectoryTracker,
)
from rocketpy.gnc.navigation.quaternion import Quaternion


def create_waypoint(
    time: float,
    x: float,
) -> Waypoint:
    return Waypoint(
        time=time,
        position=np.array(
            [x, 0.0, 0.0],
            dtype=np.float64,
        ),
        velocity=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        attitude=Quaternion(
            1.0,
            0.0,
            0.0,
            0.0,
        ),
        angular_velocity=np.zeros(
            3,
            dtype=np.float64,
        ),
    )


def test_empty_tracker() -> None:
    tracker = TrajectoryTracker()

    assert not tracker.has_trajectory()


def test_set_trajectory() -> None:
    trajectory = ReferenceTrajectory()

    tracker = TrajectoryTracker()

    tracker.set_trajectory(
        trajectory
    )

    assert tracker.has_trajectory()


def test_clear() -> None:
    trajectory = ReferenceTrajectory()

    tracker = TrajectoryTracker(
        trajectory
    )

    tracker.clear()

    assert not tracker.has_trajectory()


def test_reference_before_first() -> None:
    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(
        create_waypoint(
            0.0,
            0.0,
        )
    )

    trajectory.add_waypoint(
        create_waypoint(
            10.0,
            10.0,
        )
    )

    tracker = TrajectoryTracker(
        trajectory
    )

    waypoint = tracker.reference_at_time(
        -1.0
    )

    assert waypoint.time == 0.0


def test_reference_after_last() -> None:
    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(
        create_waypoint(
            0.0,
            0.0,
        )
    )

    trajectory.add_waypoint(
        create_waypoint(
            10.0,
            10.0,
        )
    )

    tracker = TrajectoryTracker(
        trajectory
    )

    waypoint = tracker.reference_at_time(
        20.0
    )

    assert waypoint.time == 10.0


def test_linear_position_interpolation() -> None:
    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(
        create_waypoint(
            0.0,
            0.0,
        )
    )

    trajectory.add_waypoint(
        create_waypoint(
            10.0,
            10.0,
        )
    )

    tracker = TrajectoryTracker(
        trajectory
    )

    waypoint = tracker.reference_at_time(
        5.0
    )

    assert np.allclose(
        waypoint.position,
        np.array(
            [5.0, 0.0, 0.0]
        ),
    )


def test_reset() -> None:
    trajectory = ReferenceTrajectory()

    tracker = TrajectoryTracker(
        trajectory
    )

    tracker.reset()

    assert not tracker.has_trajectory()


def test_repr() -> None:
    tracker = TrajectoryTracker()

    assert "TrajectoryTracker" in repr(
        tracker
    )


def test_invalid_type() -> None:
    tracker = TrajectoryTracker()

    with pytest.raises(TypeError):
        tracker.set_trajectory(123)