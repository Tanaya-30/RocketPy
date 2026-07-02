"""
Flight phase manager for RocketPy GNC.

Determines the current flight phase from the navigation state and
mission configuration.
"""

from __future__ import annotations

from enum import Enum

from rocketpy.gnc.interfaces import (
    GNCConfig,
    NavState,
)


class FlightPhase(str, Enum):
    """
    Supported rocket flight phases.
    """

    RAIL = "rail"
    POWERED = "powered"
    COAST = "coast"
    DESCENT = "descent"
    LANDED = "landed"


class PhaseManager:
    """
    Determine and maintain the current flight phase.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize the phase manager."""

        self._phase = FlightPhase.RAIL

    @property
    def phase(
        self,
    ) -> FlightPhase:
        """
        Current flight phase.
        """

        return self._phase

    def update(
        self,
        nav_state: NavState,
        config: GNCConfig,
    ) -> FlightPhase:
        """
        Update the flight phase.

        Parameters
        ----------
        nav_state : NavState
            Current navigation state.

        config : GNCConfig
            Global GNC configuration.

        Returns
        -------
        FlightPhase
            Current flight phase.
        """

        upward_velocity = -nav_state.vel_down

        if (
            nav_state.speed_ms
            < config.rail_clear_velocity_ms
        ):
            self._phase = FlightPhase.RAIL

        elif (
            nav_state.accel_body_z
            > config.burnout_detection_accel
        ):
            self._phase = FlightPhase.POWERED

        elif (
            upward_velocity
            > config.apogee_detection_vel_ms
        ):
            self._phase = FlightPhase.COAST

        elif nav_state.altitude_m > 1.0:
            self._phase = FlightPhase.DESCENT

        else:
            self._phase = FlightPhase.LANDED

        return self._phase

    def reset(
        self,
    ) -> None:
        """
        Reset the phase manager.
        """

        self._phase = FlightPhase.RAIL


__all__ = [
    "FlightPhase",
    "PhaseManager",
]