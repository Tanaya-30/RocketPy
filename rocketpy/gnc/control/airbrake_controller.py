"""
Air-brake controller for RocketPy GNC.

This module implements the apogee control loop using a PID controller.
The controller commands the air-brake deployment required to achieve
the desired mission apogee.
"""

from __future__ import annotations

from rocketpy.gnc.control.pid import (
    PIDConfig,
    PIDController,
)
from rocketpy.gnc.interfaces import (
    ActuatorCommand,
    ControlModuleProtocol,
    GNCConfig,
    GuidanceReference,
    NavState,
)


class AirBrakeController(
    ControlModuleProtocol,
):
    """
    PID-based air-brake controller.

    The controller regulates the predicted apogee by adjusting the
    air-brake deployment level.
    """

    def __init__(
        self,
        config: GNCConfig,
    ) -> None:
        """
        Initialize the air-brake controller.
        """

        if not isinstance(
            config,
            GNCConfig,
        ):
            raise TypeError(
                "config must be a GNCConfig."
            )
        
        self._config = config

        self._pid = PIDController(
            PIDConfig(
                kp=config.apogee_pid_kp,
                ki=config.apogee_pid_ki,
                kd=config.apogee_pid_kd,
                output_min=0.0,
                output_max=config.max_air_brake_deployment,
                integral_min=(
                    -config.apogee_pid_max_integral
                ),
                integral_max=(
                    config.apogee_pid_max_integral
                ),
            )
        )

    def compute(
        self,
        nav_state: NavState,
        guidance: GuidanceReference,
        dt: float,
    ) -> ActuatorCommand:
        """
        Compute the air-brake command.

        Parameters
        ----------
        nav_state : NavState
            Current navigation state.

        guidance : GuidanceReference
            Current guidance reference.

        dt : float
            Controller timestep in seconds.

        Returns
        -------
        ActuatorCommand
            Desired actuator command.
        """

        command = ActuatorCommand()

        if not nav_state.valid:

            command.safe_mode = True

            return command

        error = (
            guidance.estimated_apogee_m
            - guidance.target_altitude_m
        )

        deployment = self._pid.compute(
            setpoint=0.0,
            measurement=-error,
            dt=dt,
        )

        command.air_brake_deployment = deployment

        return command
    
    @property
    def pid(
        self,
    ) -> PIDController:
        """
        Return the internal PID controller.
        """

        return self._pid

    @property
    def config(
        self,
    ) -> GNCConfig:
        """
        Return the controller configuration.
        """

        return self._config

    def reset(
        self,
    ) -> None:
        """
        Reset the controller state.
        """

        self._pid.reset()

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"max_deployment="
            f"{self._config.max_air_brake_deployment}"
            ")"
        )
    
__all__ = [
    "AirBrakeController",
]