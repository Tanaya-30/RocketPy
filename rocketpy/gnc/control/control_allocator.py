"""
Control allocation for RocketPy GNC.

This module combines actuator commands produced by individual control
modules into a single command sent to the vehicle actuators.
"""

from __future__ import annotations

from rocketpy.gnc.interfaces import (
    ActuatorCommand,
)


class ControlAllocator:
    """
    Control allocator.

    Merges actuator commands from multiple controllers into a single
    actuator command.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize the allocator.
        """

        pass

    def allocate(
        self,
        *commands: ActuatorCommand,
    ) -> ActuatorCommand:
        """
        Merge multiple actuator commands.

        Parameters
        ----------
        *commands : ActuatorCommand
            Commands produced by individual controllers.

        Returns
        -------
        ActuatorCommand
            Combined actuator command.
        """

        command = ActuatorCommand()

        for controller_command in commands:

            if not isinstance(
                controller_command,
                ActuatorCommand,
            ):
                raise TypeError(
                    "commands must be ActuatorCommand objects."
                )

            command.air_brake_deployment = max(
                command.air_brake_deployment,
                controller_command.air_brake_deployment,
            )

            command.gimbal_pitch_rad = (
                controller_command.gimbal_pitch_rad
            )

            command.gimbal_yaw_rad = (
                controller_command.gimbal_yaw_rad
            )

            command.rcs_thrusters = (
                controller_command.rcs_thrusters.copy()
            )

            command.safe_mode = (
                command.safe_mode
                or controller_command.safe_mode
            )

        return command

    def reset(
        self,
    ) -> None:
        """
        Reset the allocator.
        """

        return None

    def copy(
        self,
    ) -> "ControlAllocator":
        """
        Return a copy of the allocator.
        """

        return ControlAllocator()

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        return (
            f"{self.__class__.__name__}()"
        )
    
__all__ = [
    "ControlAllocator",
]