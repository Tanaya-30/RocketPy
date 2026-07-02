"""
Unit tests for pitch_program.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocketpy.gnc.guidance.pitch_program import (
    PitchProgram,
    PitchWaypoint,
)
from rocketpy.gnc.navigation.quaternion import Quaternion


def test_waypoint_creation() -> None:
    waypoint = PitchWaypoint(
        time=0.0,
        pitch_deg=90.0,
    )

    assert waypoint.time == 0.0
    assert waypoint.pitch_deg == 90.0


def test_invalid_negative_time() -> None:
    with pytest.raises(ValueError):
        PitchWaypoint(
            time=-1.0,
            pitch_deg=90.0,
        )


def test_invalid_pitch_range() -> None:
    with pytest.raises(ValueError):
        PitchWaypoint(
            time=0.0,
            pitch_deg=181.0,
        )


def test_add_waypoint() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(0.0, 90.0)
    )

    assert len(program) == 1


def test_waypoint_order_validation() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(0.0, 90.0)
    )

    with pytest.raises(ValueError):
        program.add_waypoint(
            PitchWaypoint(0.0, 80.0)
        )


def test_duration() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(0.0, 90.0)
    )

    program.add_waypoint(
        PitchWaypoint(10.0, 45.0)
    )

    assert program.duration() == 10.0


def test_pitch_interpolation() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(0.0, 90.0)
    )

    program.add_waypoint(
        PitchWaypoint(10.0, 70.0)
    )

    assert np.isclose(
        program.pitch_at_time(5.0),
        80.0,
    )


def test_pitch_before_first_waypoint() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(2.0, 90.0)
    )

    program.add_waypoint(
        PitchWaypoint(4.0, 80.0)
    )

    assert program.pitch_at_time(0.0) == 90.0


def test_pitch_after_last_waypoint() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(0.0, 90.0)
    )

    program.add_waypoint(
        PitchWaypoint(5.0, 70.0)
    )

    assert program.pitch_at_time(100.0) == 70.0


def test_quaternion_generation() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(0.0, 90.0)
    )

    quaternion = program.quaternion_at_time(
        0.0
    )

    assert isinstance(
        quaternion,
        Quaternion,
    )


def test_copy() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(0.0, 90.0)
    )

    copied = program.copy()

    assert copied is not program
    assert len(copied) == len(program)
    assert copied[0] == program[0]


def test_clear() -> None:
    program = PitchProgram()

    program.add_waypoint(
        PitchWaypoint(0.0, 90.0)
    )

    program.clear()

    assert len(program) == 0


def test_validate_empty_program() -> None:
    program = PitchProgram()

    with pytest.raises(ValueError):
        program.validate()


def test_repr() -> None:
    program = PitchProgram()

    assert "PitchProgram" in repr(program)