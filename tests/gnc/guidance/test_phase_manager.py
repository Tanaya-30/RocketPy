"""
Unit tests for phase_manager.py.
"""

from __future__ import annotations

from rocketpy.gnc.guidance.phase_manager import (
    FlightPhase,
    PhaseManager,
)
from rocketpy.gnc.interfaces import (
    GNCConfig,
    NavState,
)


def test_initial_phase() -> None:
    manager = PhaseManager()

    assert manager.phase == FlightPhase.RAIL


def test_reset() -> None:
    manager = PhaseManager()

    nav_state = NavState(
        speed_ms=50.0,
        accel_body_z=1.0,
    )

    manager.update(
        nav_state,
        GNCConfig(),
    )

    manager.reset()

    assert manager.phase == FlightPhase.RAIL


def test_rail_phase() -> None:
    manager = PhaseManager()

    nav_state = NavState(
        speed_ms=10.0,
    )

    phase = manager.update(
        nav_state,
        GNCConfig(),
    )

    assert phase == FlightPhase.RAIL


def test_powered_phase() -> None:
    manager = PhaseManager()

    nav_state = NavState(
        speed_ms=40.0,
        accel_body_z=2.0,
    )

    phase = manager.update(
        nav_state,
        GNCConfig(),
    )

    assert phase == FlightPhase.POWERED


def test_coast_phase() -> None:
    manager = PhaseManager()

    nav_state = NavState(
        speed_ms=60.0,
        accel_body_z=0.0,
        vel_down=-100.0,
        altitude_m=1000.0,
    )

    phase = manager.update(
        nav_state,
        GNCConfig(),
    )

    assert phase == FlightPhase.COAST


def test_descent_phase() -> None:
    manager = PhaseManager()

    nav_state = NavState(
        speed_ms=60.0,
        accel_body_z=0.0,
        vel_down=5.0,
        altitude_m=500.0,
    )

    phase = manager.update(
        nav_state,
        GNCConfig(),
    )

    assert phase == FlightPhase.DESCENT


def test_landed_phase() -> None:
    manager = PhaseManager()

    nav_state = NavState(
        speed_ms=60.0,
        accel_body_z=0.0,
        vel_down=0.0,
        altitude_m=0.0,
    )

    phase = manager.update(
        nav_state,
        GNCConfig(),
    )

    assert phase == FlightPhase.LANDED


def test_phase_property() -> None:
    manager = PhaseManager()

    nav_state = NavState(
        speed_ms=60.0,
        accel_body_z=0.0,
        vel_down=0.0,
        altitude_m=0.0,
    )

    manager.update(
        nav_state,
        GNCConfig(),
    )

    assert manager.phase == FlightPhase.LANDED