"""
Unit tests for pid.py.
"""

from __future__ import annotations

import math

import pytest

from rocketpy.gnc.control.pid import (
    PIDConfig,
    PIDController,
)


def create_pid() -> PIDController:
    return PIDController(
        PIDConfig(
            kp=2.0,
            ki=0.5,
            kd=0.1,
        )
    )


def test_pid_config_creation() -> None:
    config = PIDConfig(
        kp=1.0,
        ki=2.0,
        kd=3.0,
    )

    assert config.kp == 1.0
    assert config.ki == 2.0
    assert config.kd == 3.0


def test_invalid_output_limits() -> None:
    with pytest.raises(ValueError):
        PIDConfig(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            output_min=5.0,
            output_max=1.0,
        )


def test_invalid_integral_limits() -> None:
    with pytest.raises(ValueError):
        PIDConfig(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            integral_min=10.0,
            integral_max=5.0,
        )


def test_invalid_derivative_filter() -> None:
    with pytest.raises(ValueError):
        PIDConfig(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            derivative_filter=2.0,
        )


def test_compute_returns_float() -> None:
    pid = create_pid()

    output = pid.compute(
        setpoint=10.0,
        measurement=8.0,
        dt=0.1,
    )

    assert isinstance(
        output,
        float,
    )


def test_zero_error() -> None:
    pid = create_pid()

    output = pid.compute(
        5.0,
        5.0,
        0.1,
    )

    assert math.isclose(
        output,
        0.0,
    )


def test_integral_accumulates() -> None:
    pid = create_pid()

    pid.compute(
        10.0,
        0.0,
        0.1,
    )

    first = pid.integral

    pid.compute(
        10.0,
        0.0,
        0.1,
    )

    assert pid.integral > first


def test_output_saturation() -> None:
    pid = PIDController(
        PIDConfig(
            kp=100.0,
            ki=0.0,
            kd=0.0,
            output_min=-10.0,
            output_max=10.0,
        )
    )

    output = pid.compute(
        100.0,
        0.0,
        0.1,
    )

    assert output == 10.0


def test_reset() -> None:
    pid = create_pid()

    pid.compute(
        10.0,
        0.0,
        0.1,
    )

    pid.reset()

    assert pid.integral == 0.0
    assert pid.previous_error == 0.0
    assert pid.previous_derivative == 0.0
    assert not pid.initialized


def test_copy() -> None:
    pid = create_pid()

    pid.compute(
        5.0,
        0.0,
        0.1,
    )

    copied = pid.copy()

    assert copied is not pid

    assert copied.integral == pid.integral

    assert (
        copied.previous_error
        == pid.previous_error
    )

    assert (
        copied.previous_derivative
        == pid.previous_derivative
    )


def test_invalid_dt() -> None:
    pid = create_pid()

    with pytest.raises(ValueError):
        pid.compute(
            1.0,
            0.0,
            0.0,
        )


def test_invalid_measurement() -> None:
    pid = create_pid()

    with pytest.raises(ValueError):
        pid.compute(
            1.0,
            math.nan,
            0.1,
        )


def test_invalid_setpoint() -> None:
    pid = create_pid()

    with pytest.raises(ValueError):
        pid.compute(
            math.nan,
            0.0,
            0.1,
        )


def test_repr() -> None:
    pid = create_pid()

    assert "PIDController" in repr(
        pid
    )