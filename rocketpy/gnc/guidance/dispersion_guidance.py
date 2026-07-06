"""
Dispersion-aware guidance for RocketPy GNC.

This module computes corrections to the nominal trajectory based on
predicted landing dispersion and navigation uncertainty.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.guidance.targeting import MissionTarget
from rocketpy.gnc.interfaces import NavState


class DispersionGuidance:
    """
    Dispersion-aware guidance.

    Computes trajectory correction vectors from the current navigation
    state toward the desired landing target.
    """

    def __init__(
        self,
        target: MissionTarget | None = None,
    ) -> None:
        """
        Initialize the dispersion guidance.
        """

        self._target = target

    @property
    def target(
        self,
    ) -> MissionTarget | None:
        """
        Return the active mission target.

        The returned mission target is retrieved from
        the internally stored target.
        """

        return self._target

    def set_target(
        self,
        target: MissionTarget,
    ) -> None:
        """
        Set the mission target.

        The supplied mission target is validated and
        stored as the current target.
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
        Remove the current target.
        """

        self._target = None

    def has_target(
        self,
    ) -> bool:
        """
        Return True if a target exists.
        """

        return self._target is not None

    def correction_vector(
        self,
        nav_state: NavState,
    ) -> NDArray[np.float64]:
        """
        Compute the trajectory correction vector.

        The returned correction vector is computed
        from the current navigation state and the
        stored mission target.
        """

        if self._target is None:
            raise ValueError("Mission target has not been defined.")

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

    def cross_track_error(
        self,
        nav_state: NavState,
    ) -> float:
        """
        Compute the horizontal cross-track error.

        The returned error is computed from the
        trajectory correction vector.
        """

        correction = self.correction_vector(
            nav_state,
        )

        return float(np.linalg.norm(correction[:2]))

    def along_track_error(
        self,
        nav_state: NavState,
    ) -> float:
        """
        Compute the vertical tracking error.

        The returned error is computed from the
        trajectory correction vector.
        """

        correction = self.correction_vector(
            nav_state,
        )

        return float(correction[2])

    def dispersion_radius(
        self,
        nav_state: NavState,
    ) -> float:
        """
        Estimate the landing dispersion radius.

        The returned dispersion radius is computed
        from the current horizontal error and
        vehicle altitude.
        """

        horizontal = self.cross_track_error(
            nav_state,
        )

        altitude = max(
            nav_state.altitude_m,
            1.0,
        )

        return 0.05 * horizontal + 0.01 * altitude

    def reset(
        self,
    ) -> None:
        """
        Reset the dispersion guidance.
        """

        self._target = None

    def copy(
        self,
    ) -> "DispersionGuidance":
        """
        Return a deep copy of the dispersion guidance.
        """

        if self._target is None:
            return DispersionGuidance()

        target = MissionTarget(
            target_apogee=self._target.target_apogee,
            landing_position=self._target.landing_position.copy(),
            mission_time=self._target.mission_time,
            max_acceleration=self._target.max_acceleration,
            max_dynamic_pressure=self._target.max_dynamic_pressure,
        )

        return DispersionGuidance(
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
    "DispersionGuidance",
]
