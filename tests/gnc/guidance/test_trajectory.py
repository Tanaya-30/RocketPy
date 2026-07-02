"""
Unit tests for the RocketPy GNC reference trajectory module.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocketpy.gnc.guidance.trajectory import ReferenceTrajectory, Waypoint
from rocketpy.gnc.navigation.quaternion import Quaternion


def create_waypoint(time: float) -> Waypoint:
    """
    Create a valid waypoint for testing.
    """

    return Waypoint(
        time=time,
        position=np.zeros(3, dtype=np.float64),
        velocity=np.zeros(3, dtype=np.float64),
        attitude=Quaternion.identity(),
        angular_velocity=np.zeros(3, dtype=np.float64),
    )


def test_empty_trajectory() -> None:
    """
    Test an empty trajectory.
    """

    trajectory = ReferenceTrajectory()

    assert len(trajectory) == 0
    assert trajectory.is_empty()
    assert not trajectory


def test_add_waypoint() -> None:
    """
    Test adding a waypoint.
    """

    trajectory = ReferenceTrajectory()

    waypoint = create_waypoint(0.0)

    trajectory.add_waypoint(waypoint)

    assert len(trajectory) == 1
    assert trajectory.first_waypoint() == waypoint
    assert trajectory.last_waypoint() == waypoint


def test_multiple_waypoints() -> None:
    """
    Test multiple waypoint insertion.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(0.0))
    trajectory.add_waypoint(create_waypoint(1.0))
    trajectory.add_waypoint(create_waypoint(2.0))

    assert len(trajectory) == 3
    assert trajectory.duration() == pytest.approx(2.0)


def test_waypoint_order_validation() -> None:
    """
    Waypoints must be added in increasing time order.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(1.0))

    with pytest.raises(ValueError):
        trajectory.add_waypoint(create_waypoint(0.5))


def test_insert_waypoint() -> None:
    """
    Test waypoint insertion.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(0.0))
    trajectory.add_waypoint(create_waypoint(2.0))

    trajectory.insert_waypoint(
        1,
        create_waypoint(1.0),
    )

    assert len(trajectory) == 3
    assert trajectory.get_waypoint(1).time == 1.0


def test_remove_waypoint() -> None:
    """
    Test waypoint removal.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(0.0))
    trajectory.add_waypoint(create_waypoint(1.0))

    removed = trajectory.remove_waypoint(0)

    assert removed.time == 0.0
    assert len(trajectory) == 1


def test_clear() -> None:
    """
    Test clearing a trajectory.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(0.0))
    trajectory.add_waypoint(create_waypoint(1.0))

    trajectory.clear()

    assert trajectory.is_empty()
    assert len(trajectory) == 0


def test_copy() -> None:
    """
    Test trajectory copying.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(0.0))
    trajectory.add_waypoint(create_waypoint(1.0))

    copied = trajectory.copy()

    assert copied is not trajectory
    assert len(copied) == len(trajectory)
    assert copied.duration() == trajectory.duration()


def test_iteration() -> None:
    """
    Test iteration.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(0.0))
    trajectory.add_waypoint(create_waypoint(1.0))

    times = [waypoint.time for waypoint in trajectory]

    assert times == [0.0, 1.0]


def test_getitem() -> None:
    """
    Test indexing.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(0.0))
    trajectory.add_waypoint(create_waypoint(1.0))

    assert trajectory[0].time == 0.0
    assert trajectory[1].time == 1.0


def test_contains() -> None:
    """
    Test membership.
    """

    trajectory = ReferenceTrajectory()

    waypoint = create_waypoint(0.0)

    trajectory.add_waypoint(waypoint)

    assert waypoint in trajectory


def test_waypoint_validation() -> None:
    """
    Invalid waypoint vectors should raise.
    """

    with pytest.raises(ValueError):
        Waypoint(
            time=0.0,
            position=np.zeros(2),
            velocity=np.zeros(3),
            attitude=Quaternion.identity(),
            angular_velocity=np.zeros(3),
        )


def test_negative_time() -> None:
    """
    Negative waypoint time should fail.
    """

    with pytest.raises(ValueError):
        create_waypoint(-1.0)


def test_duration_single_waypoint() -> None:
    """
    Duration with one waypoint should be zero.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(5.0))

    assert trajectory.duration() == pytest.approx(0.0)


def test_repr() -> None:
    """
    Test string representation.
    """

    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(create_waypoint(0.0))

    representation = repr(trajectory)

    assert "ReferenceTrajectory" in representation
    assert "num_waypoints=1" in representation
