"""
Unit tests for landing_guidance.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocketpy.gnc.guidance.landing_guidance import (
    LandingGuidance,
)
from rocketpy.gnc.guidance.targeting import MissionTarget
from rocketpy.gnc.interfaces import NavState


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


def test_empty_guidance() -> None:
    guidance = LandingGuidance()

    assert not guidance.has_target()


def test_set_target() -> None:
    guidance = LandingGuidance()

    target = create_target()

    guidance.set_target(target)

    assert guidance.target == target


def test_clear() -> None:
    guidance = LandingGuidance(
        create_target()
    )

    guidance.clear()

    assert not guidance.has_target()


def test_horizontal_distance() -> None:
    guidance = LandingGuidance(
        create_target()
    )

    nav = NavState(
        pos_north=0.0,
        pos_east=0.0,
    )

    distance = guidance.horizontal_distance(
        nav
    )

    assert np.isclose(
        distance,
        np.sqrt(
            100.0**2 + 50.0**2
        ),
    )


def test_direction_vector_norm() -> None:
    guidance = LandingGuidance(
        create_target()
    )

    nav = NavState(
        pos_north=0.0,
        pos_east=0.0,
    )

    direction = guidance.direction_vector(
        nav
    )

    assert np.isclose(
        np.linalg.norm(direction),
        1.0,
    )


def test_landing_error() -> None:
    guidance = LandingGuidance(
        create_target()
    )

    nav = NavState(
        pos_north=20.0,
        pos_east=10.0,
        altitude_m=0.0,
    )

    error = guidance.landing_error(
        nav
    )

    assert np.allclose(
        error,
        np.array(
            [80.0, 40.0, 0.0]
        ),
    )


def test_compute_reference() -> None:
    guidance = LandingGuidance(
        create_target()
    )

    nav = NavState(
        time=5.0,
        altitude_m=120.0,
    )

    reference = guidance.compute_reference(
        nav
    )

    assert reference.flight_phase == "descent"
    assert reference.target_altitude_m == 0.0
    assert reference.estimated_apogee_m == 120.0


def test_compute_reference_without_target() -> None:
    guidance = LandingGuidance()

    with pytest.raises(ValueError):
        guidance.compute_reference(
            NavState()
        )


def test_invalid_target() -> None:
    guidance = LandingGuidance()

    with pytest.raises(TypeError):
        guidance.set_target(123)