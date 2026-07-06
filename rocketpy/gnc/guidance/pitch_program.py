"""
Pitch program guidance for RocketPy GNC.

This module implements a pre-planned ascent pitch program.
The pitch program defines the desired vehicle attitude as a
function of mission elapsed time.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from rocketpy.gnc.navigation.quaternion import Quaternion


@dataclass(frozen=True, slots=True)
class PitchWaypoint:
    """
    Pitch program waypoint.

    Parameters
    ----------
    time : float
        Mission elapsed time in seconds.

    pitch_deg : float
        Desired pitch angle in degrees.
    """

    time: float
    pitch_deg: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValueError("time must be finite.")

        if self.time < 0.0:
            raise ValueError("time cannot be negative.")

        if not np.isfinite(self.pitch_deg):
            raise ValueError("pitch_deg must be finite.")

        if self.pitch_deg < -180.0:
            raise ValueError("pitch_deg out of range.")

        if self.pitch_deg > 180.0:
            raise ValueError("pitch_deg out of range.")


class PitchProgram:
    """
    Time-based pitch program.

    The program interpolates between user-defined pitch waypoints
    and generates the desired attitude quaternion.
    """

    def __init__(
        self,
    ) -> None:

        self._waypoints: list[PitchWaypoint] = []
        self._times: list[float] = []

    def add_waypoint(
        self,
        waypoint: PitchWaypoint,
    ) -> None:
        """
        Add a waypoint to the pitch program.
        """

        if not isinstance(
            waypoint,
            PitchWaypoint,
        ):
            raise TypeError("waypoint must be a PitchWaypoint.")

        if self._waypoints:

            if waypoint.time <= self._waypoints[-1].time:
                raise ValueError("Waypoint times must be strictly increasing.")

        self._waypoints.append(waypoint)

        self._times.append(waypoint.time)

    def clear(
        self,
    ) -> None:
        """
        Remove all waypoints.
        """

        self._waypoints.clear()
        self._times.clear()

    def validate(
        self,
    ) -> None:
        """
        Validate the pitch program.
        """

        if not self._waypoints:
            raise ValueError("Pitch program contains no waypoints.")

        previous_time = -np.inf

        for waypoint in self._waypoints:

            if waypoint.time <= previous_time:
                raise ValueError("Waypoint times must be strictly increasing.")

            previous_time = waypoint.time

    def duration(
        self,
    ) -> float:
        """
        Return the total program duration.
        """

        if not self._waypoints:
            return 0.0

        return self._waypoints[-1].time - self._waypoints[0].time

    def num_waypoints(
        self,
    ) -> int:
        """
        Return the number of stored waypoints.
        """

        return len(self._waypoints)

    def pitch_at_time(
        self,
        time: float,
    ) -> float:
        """
        Return the commanded pitch angle at the specified time.

        Linear interpolation is performed between adjacent waypoints.
        """

        self.validate()

        if time <= self._waypoints[0].time:
            return self._waypoints[0].pitch_deg

        if time >= self._waypoints[-1].time:
            return self._waypoints[-1].pitch_deg

        index = bisect_right(
            self._times,
            time,
        )
        first = self._waypoints[index - 1]
        second = self._waypoints[index]

        alpha = (time - first.time) / (second.time - first.time)

        return first.pitch_deg + alpha * (second.pitch_deg - first.pitch_deg)

    def quaternion_at_time(
        self,
        time: float,
    ) -> Quaternion:
        """
        Return the desired attitude quaternion.

        The pitch program assumes zero roll and zero yaw.
        """

        pitch = np.deg2rad(self.pitch_at_time(time))

        half_pitch = 0.5 * pitch

        return Quaternion(
            np.cos(half_pitch),
            0.0,
            np.sin(half_pitch),
            0.0,
        )

    def copy(
        self,
    ) -> "PitchProgram":
        """
        Return a deep copy of the pitch program.
        """

        program = PitchProgram()

        for waypoint in self._waypoints:
            program.add_waypoint(
                PitchWaypoint(
                    time=waypoint.time,
                    pitch_deg=waypoint.pitch_deg,
                )
            )

        return program

    def __len__(
        self,
    ) -> int:
        """
        Return the number of waypoints.
        """

        return len(self._waypoints)

    def __iter__(
        self,
    ) -> Iterator[PitchWaypoint]:
        """
        Iterate over the waypoints.
        """

        return iter(self._waypoints)

    def __getitem__(
        self,
        index: int | slice,
    ):
        """
        Return waypoint(s).
        """

        return self._waypoints[index]

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        return f"{self.__class__.__name__}" f"(num_waypoints={len(self)})"


__all__ = [
    "PitchWaypoint",
    "PitchProgram",
]
