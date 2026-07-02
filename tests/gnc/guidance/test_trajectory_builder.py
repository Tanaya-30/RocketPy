"""
Unit tests for the RocketPy GNC trajectory builder.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocketpy.gnc.guidance.trajectory import ReferenceTrajectory
from rocketpy.gnc.guidance.trajectory_builder import TrajectoryBuilder


class DummyFlight:
    """
    Simple Flight mock used for testing.
    """

    def __init__(self, solution):
        self.solution = solution


def create_state(time: float) -> np.ndarray:
    """
    Create a valid RocketPy state vector.
    """

    return np.array(
        [
            time,
            0.0,
            0.0,
            100.0,
            0.0,
            0.0,
            50.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def test_build_from_solution() -> None:
    """
    Build a trajectory from a solution.
    """

    builder = TrajectoryBuilder()

    solution = [
        create_state(0.0),
        create_state(1.0),
        create_state(2.0),
    ]

    trajectory = builder.from_solution(solution)

    assert isinstance(
        trajectory,
        ReferenceTrajectory,
    )

    assert len(trajectory) == 3


def test_build_from_flight() -> None:
    """
    Build a trajectory from a Flight object.
    """

    builder = TrajectoryBuilder()

    flight = DummyFlight(
        [
            create_state(0.0),
            create_state(1.0),
        ]
    )

    trajectory = builder.from_flight(flight)

    assert len(trajectory) == 2


def test_empty_solution() -> None:
    """
    Empty solution should fail.
    """

    builder = TrajectoryBuilder()

    with pytest.raises(ValueError):
        builder.from_solution([])


def test_invalid_state_size() -> None:
    """
    Invalid state vector length.
    """

    builder = TrajectoryBuilder()

    solution = [
        np.zeros(
            10,
            dtype=np.float64,
        )
    ]

    with pytest.raises(ValueError):
        builder.from_solution(solution)


def test_non_monotonic_time() -> None:
    """
    Time must increase.
    """

    builder = TrajectoryBuilder()

    solution = [
        create_state(1.0),
        create_state(0.5),
    ]

    with pytest.raises(ValueError):
        builder.from_solution(solution)


def test_nan_state() -> None:
    """
    NaN values are rejected.
    """

    builder = TrajectoryBuilder()

    state = create_state(0.0)

    state[4] = np.nan

    with pytest.raises(ValueError):
        builder.from_solution([state])


def test_missing_solution_attribute() -> None:
    """
    Flight must expose a solution attribute.
    """

    builder = TrajectoryBuilder()

    class InvalidFlight:
        pass

    with pytest.raises(TypeError):
        builder.from_flight(InvalidFlight())


def test_waypoint_contents() -> None:
    """
    Verify waypoint contents are copied correctly.
    """

    builder = TrajectoryBuilder()

    trajectory = builder.from_solution([create_state(0.0)])

    waypoint = trajectory.first_waypoint()

    assert waypoint.time == 0.0

    np.testing.assert_allclose(
        waypoint.position,
        [0.0, 0.0, 100.0],
    )

    np.testing.assert_allclose(
        waypoint.velocity,
        [0.0, 0.0, 50.0],
    )


def test_multiple_waypoints() -> None:
    """
    Verify multiple waypoint conversion.
    """

    builder = TrajectoryBuilder()

    solution = [
        create_state(0.0),
        create_state(0.5),
        create_state(1.0),
        create_state(1.5),
        create_state(2.0),
    ]

    trajectory = builder.from_solution(solution)

    assert len(trajectory) == 5

    assert trajectory.last_waypoint().time == 2.0
