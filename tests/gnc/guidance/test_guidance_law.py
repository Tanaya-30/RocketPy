"""
Unit tests for guidance_law.py.
"""

from __future__ import annotations

import numpy as np

from rocketpy.gnc.guidance.guidance_law import GuidanceLaw
from rocketpy.gnc.guidance.targeting import MissionTarget
from rocketpy.gnc.guidance.trajectory import (
    ReferenceTrajectory,
    Waypoint,
)
from rocketpy.gnc.interfaces import (
    GNCConfig,
    NavState,
)
from rocketpy.gnc.navigation.quaternion import Quaternion


def create_config() -> GNCConfig:
    return GNCConfig()


def create_target() -> MissionTarget:
    return MissionTarget(
        target_apogee=3000.0,
        landing_position=np.array(
            [100.0, 50.0, 0.0],
            dtype=np.float64,
        ),
        mission_time=300.0,
        max_acceleration=30.0,
        max_dynamic_pressure=50000.0,
    )


def create_waypoint(
    time: float,
) -> Waypoint:
    return Waypoint(
        time=time,
        position=np.zeros(
            3,
            dtype=np.float64,
        ),
        velocity=np.zeros(
            3,
            dtype=np.float64,
        ),
        attitude=Quaternion.identity(),
        angular_velocity=np.zeros(
            3,
            dtype=np.float64,
        ),
    )


def create_trajectory() -> ReferenceTrajectory:
    trajectory = ReferenceTrajectory()

    trajectory.add_waypoint(
        create_waypoint(0.0)
    )

    trajectory.add_waypoint(
        create_waypoint(5.0)
    )

    return trajectory


def test_initial_state() -> None:
    law = GuidanceLaw(
        create_config()
    )

    assert law.target is None
    assert law.phase.value == "rail"


def test_set_target() -> None:
    law = GuidanceLaw(
        create_config()
    )

    target = create_target()

    law.set_target(target)

    assert law.target is target


def test_clear_target() -> None:
    law = GuidanceLaw(
        create_config()
    )

    law.set_target(
        create_target()
    )

    law.clear_target()

    assert law.target is None


def test_set_reference_trajectory() -> None:
    law = GuidanceLaw(
        create_config()
    )

    law.set_reference_trajectory(
        create_trajectory()
    )

    assert (
        law.trajectory_tracker.has_trajectory()
    )


def test_clear_reference_trajectory() -> None:
    law = GuidanceLaw(
        create_config()
    )

    law.set_reference_trajectory(
        create_trajectory()
    )

    law.clear_reference_trajectory()

    assert (
        not law.trajectory_tracker.has_trajectory()
    )


def test_compute_reference() -> None:
    law = GuidanceLaw(
        create_config()
    )

    nav = NavState(
        time=10.0,
        altitude_m=100.0,
    )

    reference = law.compute_reference(
        nav,
        create_config(),
    )

    assert reference.time == 10.0

    assert (
        reference.target_altitude_m
        == create_config().target_apogee_m
    )


def test_compute_reference_with_target() -> None:
    law = GuidanceLaw(
        create_config()
    )

    law.set_target(
        create_target()
    )

    reference = law.compute_reference(
        NavState(),
        create_config(),
    )

    assert (
        reference.target_altitude_m
        == 3000.0
    )


def test_reset() -> None:
    law = GuidanceLaw(
        create_config()
    )

    law.set_target(
        create_target()
    )

    law.set_reference_trajectory(
        create_trajectory()
    )

    law.reset()

    assert law.target is None

    assert (
        not law.trajectory_tracker.has_trajectory()
    )


def test_repr() -> None:
    law = GuidanceLaw(
        create_config()
    )

    assert "GuidanceLaw" in repr(
        law
    )


def test_phase_property() -> None:
    law = GuidanceLaw(
        create_config()
    )

    assert law.phase.value == "rail"