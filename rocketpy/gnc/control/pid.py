"""
Production PID controller for RocketPy GNC.

This module implements a reusable PID controller that can be used by
the air-brake, TVC, roll, pitch, and yaw controllers.
"""

from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass(slots=True)
class PIDConfig:
    """
    Configuration parameters for a PID controller.
    """

    kp: float
    ki: float
    kd: float

    output_min: float = -math.inf
    output_max: float = math.inf

    integral_min: float = -math.inf
    integral_max: float = math.inf

    derivative_filter: float = 1.0

    def __post_init__(self) -> None:

        for value in (
            self.kp,
            self.ki,
            self.kd,
            self.output_min,
            self.output_max,
            self.integral_min,
            self.integral_max,
            self.derivative_filter,
        ):

            if not math.isfinite(value):

                if value not in (
                    math.inf,
                    -math.inf,
                ):
                    raise ValueError(
                        "PID parameters must be finite."
                    )

        if self.output_min > self.output_max:
            raise ValueError(
                "output_min must not exceed output_max."
            )

        if self.integral_min > self.integral_max:
            raise ValueError(
                "integral_min must not exceed integral_max."
            )

        if not (
            0.0 < self.derivative_filter <= 1.0
        ):
            raise ValueError(
                "derivative_filter must be in (0, 1]."
            )


class PIDController:
    """
    Generic PID controller.

    This implementation supports

    - variable timestep
    - integral anti-windup
    - output saturation
    - derivative filtering
    """

    def __init__(
        self,
        config: PIDConfig,
    ) -> None:

        if not isinstance(
            config,
            PIDConfig,
        ):
            raise TypeError(
                "config must be a PIDConfig."
            )

        self._config = config

        self._integral = 0.0

        self._previous_error = 0.0

        self._previous_derivative = 0.0

        self._initialized = False

    def compute(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
    ) -> float:
        """
        Compute the PID controller output.

        Parameters
        ----------
        setpoint : float
            Desired value.

        measurement : float
            Measured value.

        dt : float
            Time step in seconds.

        Returns
        -------
        float
            Saturated controller output.
        """

        if not math.isfinite(setpoint):
            raise ValueError(
                "setpoint must be finite."
            )

        if not math.isfinite(measurement):
            raise ValueError(
                "measurement must be finite."
            )

        if not math.isfinite(dt):
            raise ValueError(
                "dt must be finite."
            )

        if dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        error = (
            setpoint
            - measurement
        )

        proportional = (
            self._config.kp
            * error
        )

        self._integral += (
            error * dt
        )

        self._integral = min(
            self._config.integral_max,
            max(
                self._config.integral_min,
                self._integral,
            ),
        )

        integral = (
            self._config.ki
            * self._integral
        )

        if not self._initialized:

            derivative = 0.0

            self._initialized = True

        else:

            derivative = (
                error
                - self._previous_error
            ) / dt

        derivative = (
            self._config.derivative_filter
            * derivative
            + (
                1.0
                - self._config.derivative_filter
            )
            * self._previous_derivative
        )

        derivative_term = (
            self._config.kd
            * derivative
        )

        output = (
            proportional
            + integral
            + derivative_term
        )

        if not math.isfinite(output):
            raise FloatingPointError(
                "PID output became non-finite."
            )


        output = min(
            self._config.output_max,
            max(
                self._config.output_min,
                output,
            ),
        )

        self._previous_error = error

        self._previous_derivative = derivative

        return output
    
    @property
    def config(
        self,
    ) -> PIDConfig:
        """
        Return the PID configuration.
        """

        return self._config

    @property
    def integral(
        self,
    ) -> float:
        """
        Return the integral state.
        """

        return self._integral

    @property
    def previous_error(
        self,
    ) -> float:
        """
        Return the previous control error.
        """

        return self._previous_error

    @property
    def initialized(
        self,
    ) -> bool:
        """
        Return True if the controller has been initialized.
        """

        return self._initialized
    
    @property
    def previous_derivative(
        self,
    ) -> float:
        """
        Return the previous derivative estimate.
        """

        return self._previous_derivative

    def reset(
        self,
    ) -> None:
        """
        Reset the controller state.
        """

        self._integral = 0.0

        self._previous_error = 0.0

        self._previous_derivative = 0.0

        self._initialized = False

    def copy(
        self,
    ) -> "PIDController":
        """
        Return a deep copy of the controller.
        """

        controller = PIDController(
            PIDConfig(
                kp=self._config.kp,
                ki=self._config.ki,
                kd=self._config.kd,
                output_min=self._config.output_min,
                output_max=self._config.output_max,
                integral_min=self._config.integral_min,
                integral_max=self._config.integral_max,
                derivative_filter=self._config.derivative_filter,
            )
        )

        controller._integral = (
            self._integral
        )

        controller._previous_error = (
            self._previous_error
        )

        controller._previous_derivative = (
            self._previous_derivative
        )

        controller._initialized = (
            self._initialized
        )

        return controller

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"kp={self._config.kp}, "
            f"ki={self._config.ki}, "
            f"kd={self._config.kd}"
            ")"
        )
    
__all__ = [
    "PIDConfig",
    "PIDController",
]