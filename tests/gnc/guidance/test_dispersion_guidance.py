"""
Unit tests for dispersion_guidance.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocketpy.gnc.guidance.dispersion_guidance import (
    DispersionGuidance,
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
    guidance = DispersionGuidance()

    assert not guidance.has_target()


def test_set_target() -> None:
    guidance = DispersionGuidance()

    target = create_target()

    guidance.set_target(target)

    assert guidance.target == target


def test_clear() -> None:
    guidance = DispersionGuidance(
        create_target()
    )

    guidance.clear()

    assert not guidance.has_target()


def test_correction_vector() -> None:
    guidance = DispersionGuidance(
        create_target()
    )

    nav = NavState(
        pos_north=20.0,
        pos_east=10.0,
        altitude_m=0.0,
    )

    correction = guidance.correction_vector(
        nav
    )

    assert np.allclose(
        correction,
        np.array(
            [80.0, 40.0, 0.0]
        ),
    )


def test_cross_track_error() -> None:
    guidance = DispersionGuidance(
        create_target()
    )

    nav = NavState()

    error = guidance.cross_track_error(
        nav
    )

    assert np.isclose(
        error,
        np.sqrt(
            100.0**2 + 50.0**2
        ),
    )


def test_along_track_error() -> None:
    guidance = DispersionGuidance(
        create_target()
    )

    nav = NavState(
        altitude_m=25.0,
    )

    error = guidance.along_track_error(
        nav
    )

    assert error == 25.0


def test_dispersion_radius() -> None:
    guidance = DispersionGuidance(
        create_target()
    )

    nav = NavState(
        altitude_m=100.0,
    )

    radius = guidance.dispersion_radius(
        nav
    )

    assert radius > 0.0


def test_reset() -> None:
    guidance = DispersionGuidance(
        create_target()
    )

    guidance.reset()

    assert not guidance.has_target()


def test_copy() -> None:
    guidance = DispersionGuidance(
        create_target()
    )

    copied = guidance.copy()

    assert copied is not guidance
    assert copied.target is not guidance.target

    assert copied.target.target_apogee == guidance.target.target_apogee

    assert np.array_equal(
        copied.target.landing_position,
        guidance.target.landing_position,
    )

    assert (
        copied.target.mission_time
        == guidance.target.mission_time
    )

    assert (
        copied.target.max_acceleration
        == guidance.target.max_acceleration
    )

    assert (
        copied.target.max_dynamic_pressure
        == guidance.target.max_dynamic_pressure
    )


def test_invalid_target() -> None:
    guidance = DispersionGuidance()

    with pytest.raises(TypeError):
        guidance.set_target(123)


def test_missing_target() -> None:
    guidance = DispersionGuidance()

    with pytest.raises(ValueError):
        guidance.correction_vector(
            NavState()
        )


def test_repr() -> None:
    guidance = DispersionGuidance()

    assert "DispersionGuidance" in repr(
        guidance
    )