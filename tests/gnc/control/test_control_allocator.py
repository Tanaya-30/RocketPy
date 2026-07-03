"""
Unit tests for control_allocator.py.
"""

from __future__ import annotations

import pytest

from rocketpy.gnc.control.control_allocator import (
    ControlAllocator,
)
from rocketpy.gnc.interfaces import (
    ActuatorCommand,
)


def create_command() -> ActuatorCommand:
    command = ActuatorCommand()

    command.air_brake_deployment = 0.5
    command.gimbal_pitch_rad = 0.1
    command.gimbal_yaw_rad = -0.1
    command.rcs_thrusters = []

    return command


def test_allocator_creation() -> None:

    allocator = ControlAllocator()

    assert allocator is not None


def test_allocate_single_command() -> None:

    allocator = ControlAllocator()

    command = allocator.allocate(
        create_command(),
    )

    assert command.air_brake_deployment == 0.5

    assert command.gimbal_pitch_rad == 0.1

    assert command.gimbal_yaw_rad == -0.1

    assert command.rcs_thrusters == []


def test_allocate_multiple_commands() -> None:

    allocator = ControlAllocator()

    first = create_command()

    second = ActuatorCommand()

    second.air_brake_deployment = 0.8

    command = allocator.allocate(
        first,
        second,
    )

    assert command.air_brake_deployment == 0.8


def test_safe_mode() -> None:

    allocator = ControlAllocator()

    first = create_command()

    second = ActuatorCommand()

    second.safe_mode = True

    command = allocator.allocate(
        first,
        second,
    )

    assert command.safe_mode


def test_invalid_command() -> None:

    allocator = ControlAllocator()

    with pytest.raises(TypeError):

        allocator.allocate(
            None,
        )


def test_copy() -> None:

    allocator = ControlAllocator()

    copied = allocator.copy()

    assert copied is not allocator


def test_reset() -> None:

    allocator = ControlAllocator()

    allocator.reset()


def test_repr() -> None:

    allocator = ControlAllocator()

    assert (
        "ControlAllocator"
        in repr(
            allocator,
        )
    )