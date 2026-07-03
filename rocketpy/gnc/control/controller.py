"""
Master controller for RocketPy GNC.

This module coordinates the individual control modules and produces
the final actuator command for the vehicle.
"""

from __future__ import annotations

from rocketpy.gnc.control.airbrake_controller import (
    AirBrakeController,
)
from rocketpy.gnc.interfaces import (
    ActuatorCommand,
    ControlModuleProtocol,
    GNCConfig,
    GuidanceReference,
    NavState,
)


class Controller(
    ControlModuleProtocol,
):
    """
    Master flight controller.

    Coordinates all control modules and combines their outputs into
    a single actuator command.
    """

    def __init__(
        self,
        config: GNCConfig,
    ) -> None:
        """
        Initialize the controller.
        """

        if not isinstance(
            config,
            GNCConfig,
        ):
            raise TypeError(
                "config must be a GNCConfig."
            )

        self._config = config

        self._airbrake_controller = (
            AirBrakeController(
                config,
            )
        )

    def compute(
        self,
        nav_state: NavState,
        guidance: GuidanceReference,
        dt: float,
    ) -> ActuatorCommand:
        """
        Compute the actuator command.

        Parameters
        ----------
        nav_state : NavState
            Current navigation state.

        guidance : GuidanceReference
            Desired guidance reference.

        dt : float
            Controller timestep.

        Returns
        -------
        ActuatorCommand
            Combined actuator command.
        """

        if not isinstance(
            nav_state,
            NavState,
        ):
            raise TypeError(
                "nav_state must be a NavState."
            )

        if not isinstance(
            guidance,
            GuidanceReference,
        ):
            raise TypeError(
                "guidance must be a GuidanceReference."
            )
        
        if dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        command = ActuatorCommand()

        if not nav_state.valid:

            command.safe_mode = True

            return command

        airbrake_command = (
            self._airbrake_controller.compute(
                nav_state,
                guidance,
                dt,
            )
        )

        command.air_brake_deployment = (
            airbrake_command.air_brake_deployment
        )

        command.safe_mode = (
            airbrake_command.safe_mode
        )

        return command
    
    @property
    def config(
        self,
    ) -> GNCConfig:
        """
        Return the controller configuration.
        """

        return self._config

    @property
    def airbrake_controller(
        self,
    ) -> AirBrakeController:
        """
        Return the air-brake controller.
        """

        return self._airbrake_controller

    def reset(
        self,
    ) -> None:
        """
        Reset all control modules.
        """

        self._airbrake_controller.reset()

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"airbrake=True"
            ")"
        )
    
__all__ = [
    "Controller",
]