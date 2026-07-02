"""
Unit tests for RocketPy GNC mission targeting.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocketpy.gnc.guidance.targeting import (
    MissionTarget,
    Targeting,
)


def create_target() -> MissionTarget:
    """
    Create a valid mission target for testing.
    """

    return MissionTarget(
        target_apogee=1000.0,
        landing_position=np.array(
            [100.0, 50.0, 0.0],
            dtype=np.float64,
        ),
        mission_time=120.0,
        max_acceleration=25.0,
        max_dynamic_pressure=50000.0,
    )


def test_create_target() -> None:
    """
    Test MissionTarget creation.
    """

    target = create_target()

    assert target.target_apogee == 1000.0
    assert target.mission_time == 120.0
    assert target.max_acceleration == 25.0
    assert target.max_dynamic_pressure == 50000.0

    np.testing.assert_allclose(
        target.landing_position,
        [100.0, 50.0, 0.0],
    )


def test_invalid_landing_position() -> None:
    """
    Landing position must contain three elements.
    """

    with pytest.raises(ValueError):
        MissionTarget(
            target_apogee=1000.0,
            landing_position=np.zeros(2),
            mission_time=120.0,
            max_acceleration=25.0,
            max_dynamic_pressure=50000.0,
        )


def test_negative_apogee() -> None:
    """
    Apogee must be positive.
    """

    with pytest.raises(ValueError):
        MissionTarget(
            target_apogee=-1.0,
            landing_position=np.zeros(
                3,
                dtype=np.float64,
            ),
            mission_time=120.0,
            max_acceleration=25.0,
            max_dynamic_pressure=50000.0,
        )


def test_negative_mission_time() -> None:
    """
    Mission time must be positive.
    """

    with pytest.raises(ValueError):
        MissionTarget(
            target_apogee=1000.0,
            landing_position=np.zeros(
                3,
                dtype=np.float64,
            ),
            mission_time=-5.0,
            max_acceleration=25.0,
            max_dynamic_pressure=50000.0,
        )


def test_build_target() -> None:
    """
    Build a mission target.
    """

    targeting = Targeting()

    target = targeting.build_target(
        target_apogee=1200.0,
        landing_position=[0.0, 0.0, 0.0],
        mission_time=150.0,
        max_acceleration=30.0,
        max_dynamic_pressure=60000.0,
    )

    assert target.target_apogee == 1200.0
    assert targeting.target is target


def test_validate() -> None:
    """
    Validation should fail when no target exists.
    """

    targeting = Targeting()

    with pytest.raises(ValueError):
        targeting.validate()


def test_clear() -> None:
    """
    Test clearing the target.
    """

    targeting = Targeting()

    targeting.build_target(
        target_apogee=1000.0,
        landing_position=[0.0, 0.0, 0.0],
        mission_time=120.0,
        max_acceleration=25.0,
        max_dynamic_pressure=50000.0,
    )

    targeting.clear()

    assert targeting.target is None


def test_copy() -> None:
    """
    Test copying a mission target.
    """

    targeting = Targeting()

    targeting.build_target(
        target_apogee=1500.0,
        landing_position=[1.0, 2.0, 3.0],
        mission_time=200.0,
        max_acceleration=35.0,
        max_dynamic_pressure=70000.0,
    )

    copied = targeting.copy()

    assert copied is not targeting.target
    assert copied.target_apogee == 1500.0

    np.testing.assert_allclose(
        copied.landing_position,
        [1.0, 2.0, 3.0],
    )


def test_as_dict() -> None:
    """
    Test dictionary serialization.
    """

    targeting = Targeting()

    targeting.build_target(
        target_apogee=2000.0,
        landing_position=[5.0, 10.0, 0.0],
        mission_time=250.0,
        max_acceleration=40.0,
        max_dynamic_pressure=80000.0,
    )

    data = targeting.as_dict()

    assert data["target_apogee"] == 2000.0
    assert data["mission_time"] == 250.0
    assert data["max_acceleration"] == 40.0
    assert data["max_dynamic_pressure"] == 80000.0

    np.testing.assert_allclose(
        data["landing_position"],
        [5.0, 10.0, 0.0],
    )


def test_from_dict() -> None:
    """
    Test construction from a dictionary.
    """

    data = {
        "target_apogee": 1800.0,
        "landing_position": [20.0, 30.0, 0.0],
        "mission_time": 180.0,
        "max_acceleration": 32.0,
        "max_dynamic_pressure": 65000.0,
    }

    targeting = Targeting.from_dict(data)

    assert targeting.target is not None
    assert targeting.target.target_apogee == 1800.0

    np.testing.assert_allclose(
        targeting.target.landing_position,
        [20.0, 30.0, 0.0],
    )


def test_copy_without_target() -> None:
    """
    Copy should fail if no target exists.
    """

    targeting = Targeting()

    with pytest.raises(ValueError):
        targeting.copy()


def test_as_dict_without_target() -> None:
    """
    Serialization should fail if no target exists.
    """

    targeting = Targeting()

    with pytest.raises(ValueError):
        targeting.as_dict()