"""
Unit tests for controller.py.
"""

from __future__ import annotations

import pytest

from rocketpy.gnc.control.controller import (
    Controller,
)
from rocketpy.gnc.interfaces import (
    GNCConfig,
    GuidanceReference,
    NavState,
)


def create_config() -> GNCConfig:
    config = GNCConfig()

    config.apogee_pid_kp = 1.0
    config.apogee_pid_ki = 0.1
    config.apogee_pid_kd = 0.0

    config.apogee_pid_max_integral = 100.0

    config.max_air_brake_deployment = 1.0

    return config


def create_nav_state() -> NavState:
    nav = NavState()

    nav.valid = True

    return nav


def create_guidance() -> GuidanceReference:
    guidance = GuidanceReference()

    guidance.target_altitude_m = 3000.0
    guidance.estimated_apogee_m = 3200.0

    return guidance


def test_controller_creation() -> None:

    controller = Controller(
        create_config()
    )

    assert controller.config is not None


def test_invalid_config() -> None:

    with pytest.raises(TypeError):

        Controller(
            None,
        )


def test_compute_returns_command() -> None:

    controller = Controller(
        create_config()
    )

    command = controller.compute(
        create_nav_state(),
        create_guidance(),
        0.1,
    )

    assert command is not None


def test_safe_mode() -> None:

    controller = Controller(
        create_config()
    )

    nav = create_nav_state()

    nav.valid = False

    command = controller.compute(
        nav,
        create_guidance(),
        0.1,
    )

    assert command.safe_mode


def test_invalid_dt() -> None:

    controller = Controller(
        create_config()
    )

    with pytest.raises(ValueError):

        controller.compute(
            create_nav_state(),
            create_guidance(),
            0.0,
        )


def test_reset() -> None:

    controller = Controller(
        create_config()
    )

    controller.compute(
        create_nav_state(),
        create_guidance(),
        0.1,
    )

    controller.reset()

    assert (
        controller.airbrake_controller.pid.integral
        == 0.0
    )


def test_airbrake_controller_property() -> None:

    controller = Controller(
        create_config()
    )

    assert (
        controller.airbrake_controller
        is not None
    )


def test_repr() -> None:

    controller = Controller(
        create_config()
    )

    assert "Controller" in repr(
        controller
    )