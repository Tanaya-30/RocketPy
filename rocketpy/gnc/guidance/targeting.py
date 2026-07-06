"""
Mission targeting for RocketPy GNC.

This module defines the desired mission objectives used by the Guidance
subsystem. A MissionTarget specifies the high-level goals of a flight,
such as the desired apogee, landing location, and operational
constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MissionTarget:
    """
    Immutable mission objectives.

    The mission target stores the desired flight
    objectives used by the Guidance subsystem.


    Parameters
    ----------
    target_apogee : float
        Desired apogee in meters.

    landing_position : ndarray
        Desired landing position in the navigation frame.

    mission_time : float
        Expected mission duration in seconds.

    max_acceleration : float
        Maximum allowable acceleration in m/s².

    max_dynamic_pressure : float
        Maximum allowable dynamic pressure in Pascals.
    """

    target_apogee: float
    landing_position: NDArray[np.float64]
    mission_time: float
    max_acceleration: float
    max_dynamic_pressure: float

    def __post_init__(self) -> None:
        """
        Validate the mission target.

        The supplied mission parameters are validated
        before the target is accepted.
        """

        landing_position = np.asarray(
            self.landing_position,
            dtype=np.float64,
        )

        object.__setattr__(
            self,
            "landing_position",
            landing_position,
        )

        if self.landing_position.shape != (3,):
            raise ValueError("landing_position must have shape (3,).")

        if not np.all(np.isfinite(self.landing_position)):
            raise ValueError("landing_position contains non-finite values.")

        for value, name in (
            (self.target_apogee, "target_apogee"),
            (self.mission_time, "mission_time"),
            (self.max_acceleration, "max_acceleration"),
            (
                self.max_dynamic_pressure,
                "max_dynamic_pressure",
            ),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")


class Targeting:
    """
    Mission target manager.

    The class creates, stores, validates and
    provides access to the current mission target.
    """

    def __init__(
        self,
        target: MissionTarget | None = None,
    ) -> None:
        """
        Initialize the targeting module.

        Parameters
        ----------
        target : MissionTarget, optional
            Initial mission target.
        """

        self._target = target

    @property
    def target(
        self,
    ) -> MissionTarget | None:
        """
        Return the current mission target.

        The returned target is retrieved from the
        internally stored mission target.
        """

        return self._target

    def build_target(
        self,
        target_apogee: float,
        landing_position: (
            NDArray[np.float64] | list[float] | tuple[float, float, float]
        ),
        mission_time: float,
        max_acceleration: float,
        max_dynamic_pressure: float,
    ) -> MissionTarget:
        """
        Create and store a mission target.

        The supplied mission parameters are validated
        and stored as the current mission target.
        """

        target = MissionTarget(
            target_apogee=target_apogee,
            landing_position=landing_position,
            mission_time=mission_time,
            max_acceleration=max_acceleration,
            max_dynamic_pressure=max_dynamic_pressure,
        )

        self._target = target

        return target

    def validate(
        self,
    ) -> None:
        """
        Validate the current mission target.

        Ensures that a mission target has been defined
        before it is accessed.
        """

        if self._target is None:
            raise ValueError("No mission target has been defined.")

    def clear(
        self,
    ) -> None:
        """
        Clear the current mission target.
        """

        self._target = None

    def copy(
        self,
    ) -> MissionTarget:
        """
        Return a copy of the current mission target.
        """

        self.validate()

        assert self._target is not None

        return MissionTarget(
            target_apogee=self._target.target_apogee,
            landing_position=self._target.landing_position.copy(),
            mission_time=self._target.mission_time,
            max_acceleration=self._target.max_acceleration,
            max_dynamic_pressure=self._target.max_dynamic_pressure,
        )

    def as_dict(
        self,
    ) -> dict[str, object]:
        """
        Create a Targeting instance from a dictionary.

        The mission target is constructed from the
        supplied dictionary and stored in the new
        Targeting instance.
        """

        self.validate()

        assert self._target is not None

        return {
            "target_apogee": self._target.target_apogee,
            "landing_position": self._target.landing_position.copy(),
            "mission_time": self._target.mission_time,
            "max_acceleration": self._target.max_acceleration,
            "max_dynamic_pressure": self._target.max_dynamic_pressure,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "Targeting":
        """
        Create a Targeting instance from a dictionary.

        Parameters
        ----------
        data : dict[str, object]
            Dictionary containing mission target parameters.
        """

        target = MissionTarget(
            target_apogee=float(data["target_apogee"]),
            landing_position=np.asarray(
                data["landing_position"],
                dtype=np.float64,
            ),
            mission_time=float(data["mission_time"]),
            max_acceleration=float(data["max_acceleration"]),
            max_dynamic_pressure=float(data["max_dynamic_pressure"]),
        )

        return cls(target)


__all__ = [
    "MissionTarget",
    "Targeting",
]
