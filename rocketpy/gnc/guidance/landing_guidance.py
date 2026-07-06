"""
Landing guidance for RocketPy GNC.

This module computes guidance commands that steer the vehicle toward
a desired landing location during the descent phase.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.interfaces import (
    GuidanceReference,
    NavState,
)
from rocketpy.gnc.guidance.targeting import MissionTarget


class LandingGuidance:
    """
    Landing guidance module.

    Computes the desired landing reference from the current vehicle
    state and mission target.
    """

    def __init__(
        self,
        target: MissionTarget | None = None,
    ) -> None:
        """
        Return the current landing target.

        The returned mission target is retrieved from
        the internally stored landing target.
        """

        self._target = target

    @property
    def target(
        self,
    ) -> MissionTarget | None:
        """
        Return the current landing target.
        """

        return self._target

    def set_target(
        self,
        target: MissionTarget,
    ) -> None:
        """
        Set the landing target.

        The supplied mission target is validated and
        stored as the current landing target.
        """

        if not isinstance(
            target,
            MissionTarget,
        ):
            raise TypeError("target must be a MissionTarget.")

        self._target = target

    def clear(
        self,
    ) -> None:
        """
        Remove the landing target.
        """

        self._target = None

    def has_target(
        self,
    ) -> bool:
        """
        Return True if a landing target exists.
        """

        return self._target is not None

    def compute_reference(
        self,
        nav_state: NavState,
    ) -> GuidanceReference:
        """
        Compute the landing guidance reference.

        The returned guidance reference is computed
        from the current navigation state and the
        stored mission target.


        Parameters
        ----------
        nav_state : NavState
            Current navigation state.

        Returns
        -------
        GuidanceReference
            Landing guidance reference computed for
            the current navigation state.
        """

        if self._target is None:
            raise ValueError("Landing target has not been defined.")

        distance = self.horizontal_distance(
            nav_state,
        )

        direction = self.direction_vector(
            nav_state,
        )

        reference = GuidanceReference(
            time=nav_state.time,
        )

        reference.flight_phase = "descent"

        reference.target_altitude_m = 0.0

        reference.estimated_apogee_m = nav_state.altitude_m

        return reference

    def horizontal_distance(
        self,
        nav_state: NavState,
    ) -> float:
        """
        Return the horizontal distance to the
        landing target.

        The returned distance is computed from the
        current navigation position and the stored
        landing target.
        """

        assert self._target is not None

        current = np.array(
            [
                nav_state.pos_north,
                nav_state.pos_east,
            ],
            dtype=np.float64,
        )

        target = self._target.landing_position[:2]

        return float(np.linalg.norm(target - current))

    def direction_vector(
        self,
        nav_state: NavState,
    ) -> NDArray[np.float64]:
        """
        Return the direction toward the landing target.

        The returned unit vector is computed from the
        current navigation position and the stored
        landing target.
        """

        assert self._target is not None

        direction = self._target.landing_position[:2] - np.array(
            [
                nav_state.pos_north,
                nav_state.pos_east,
            ],
            dtype=np.float64,
        )

        norm = np.linalg.norm(direction)

        if norm < 1e-12:
            return np.zeros(
                2,
                dtype=np.float64,
            )

        return direction / norm

    def landing_error(
        self,
        nav_state: NavState,
    ) -> NDArray[np.float64]:
        """
        Return the landing position error.

        The returned position error is computed from
        the current navigation position and the
        stored landing target.

        Returns
        -------
        ndarray
            Position error [north, east, down] in meters.
        """

        assert self._target is not None

        target = self._target.landing_position

        current = np.array(
            [
                nav_state.pos_north,
                nav_state.pos_east,
                -nav_state.altitude_m,
            ],
            dtype=np.float64,
        )

        return target - current

    def reset(
        self,
    ) -> None:
        """
        Reset the landing guidance.
        """

        self._target = None

    def copy(
        self,
    ) -> "LandingGuidance":
        """
        Return a deep copy of the landing guidance.
        """

        if self._target is None:
            return LandingGuidance()

        target = MissionTarget(
            target_apogee=self._target.target_apogee,
            landing_position=self._target.landing_position.copy(),
            mission_time=self._target.mission_time,
            max_acceleration=self._target.max_acceleration,
            max_dynamic_pressure=self._target.max_dynamic_pressure,
        )

        return LandingGuidance(
            target=target,
        )

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        if self._target is None:
            return f"{self.__class__.__name__}" "(target=None)"

        return (
            f"{self.__class__.__name__}"
            "("
            f"landing_position="
            f"{self._target.landing_position}, "
            f"target_apogee="
            f"{self._target.target_apogee}"
            ")"
        )


__all__ = [
    "LandingGuidance",
]
