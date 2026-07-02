"""
Reference trajectory data structures for RocketPy GNC.

This module defines the immutable waypoint representation and the
reference trajectory container used by the Guidance subsystem.

A ReferenceTrajectory is an ordered sequence of Waypoints describing
the desired vehicle state throughout a mission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.navigation.quaternion import Quaternion


@dataclass(frozen=True, slots=True)
class Waypoint:
    """
    Immutable reference waypoint.

    Parameters
    ----------
    time : float
        Mission elapsed time in seconds.

    position : ndarray
        Desired NED position [x, y, z] in meters.

    velocity : ndarray
        Desired NED velocity [vx, vy, vz] in m/s.

    attitude : Quaternion
        Desired body-to-navigation quaternion.

    angular_velocity : ndarray
        Desired body angular velocity [wx, wy, wz] in rad/s.
    """

    time: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    attitude: Quaternion
    angular_velocity: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate waypoint contents."""

        if not np.isfinite(self.time):
            raise ValueError("Waypoint time must be finite.")

        if self.time < 0.0:
            raise ValueError("Waypoint time cannot be negative.")

        object.__setattr__(
            self,
            "position",
            np.asarray(
                self.position,
                dtype=np.float64,
            ),
        )

        object.__setattr__(
            self,
            "velocity",
            np.asarray(self.velocity, dtype=np.float64),
        )

        object.__setattr__(
            self,
            "angular_velocity",
            np.asarray(
                self.angular_velocity,
                dtype=np.float64,
            ),
        )

        self._validate_vector(
            self.position,
            "position",
        )

        self._validate_vector(
            self.velocity,
            "velocity",
        )

        self._validate_vector(
            self.angular_velocity,
            "angular_velocity",
        )

        if not isinstance(
            self.attitude,
            Quaternion,
        ):
            raise TypeError("attitude must be a Quaternion.")
        normalized = self.attitude.copy()
        normalized.normalize()

        object.__setattr__(
            self,
            "attitude",
            normalized,
        )

    @staticmethod
    def _validate_vector(
        vector: NDArray[np.float64],
        name: str,
    ) -> None:
        """
        Validate a 3D vector.
        """

        vector = np.asarray(
            vector,
            dtype=np.float64,
        )

        if vector.shape != (3,):
            raise ValueError(f"{name} must have shape (3,).")

        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} contains non-finite values.")

    def copy(self) -> "Waypoint":
        """
        Return a deep copy of the waypoint.
        """

        return Waypoint(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=self.attitude.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )

    def as_dict(self) -> dict[str, object]:
        """
        Return the waypoint as a dictionary.
        """

        return {
            "time": self.time,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "attitude": self.attitude.copy(),
            "angular_velocity": self.angular_velocity.copy(),
        }


class ReferenceTrajectory:
    """
    Ordered collection of reference waypoints.

    A ReferenceTrajectory represents the desired vehicle state over time.
    Waypoints are stored in strictly increasing time order.
    """

    def __init__(
        self,
        waypoints: list[Waypoint] | None = None,
    ) -> None:
        """
        Initialize the trajectory.

        Parameters
        ----------
        waypoints : list[Waypoint], optional
            Initial waypoint sequence.
        """

        self._waypoints: list[Waypoint] = []

        if waypoints is not None:
            for waypoint in waypoints:
                self.add_waypoint(waypoint)

    def add_waypoint(
        self,
        waypoint: Waypoint,
    ) -> None:
        """
        Append a waypoint.

        Parameters
        ----------
        waypoint : Waypoint
            Waypoint to append.
        """

        if not isinstance(
            waypoint,
            Waypoint,
        ):
            raise TypeError("waypoint must be a Waypoint.")

        if self._waypoints:

            if waypoint.time <= self._waypoints[-1].time:
                raise ValueError("Waypoint times must be strictly increasing.")

        self._waypoints.append(waypoint)

    def insert_waypoint(
        self,
        index: int,
        waypoint: Waypoint,
    ) -> None:
        """
        Insert a waypoint.

        The trajectory is validated after insertion.
        """

        if not isinstance(
            waypoint,
            Waypoint,
        ):
            raise TypeError("waypoint must be a Waypoint.")

        candidate = self._waypoints.copy()

        candidate.insert(
            index,
            waypoint,
        )

        previous_time = -np.inf

        for current_waypoint in candidate:

            if current_waypoint.time <= previous_time:
                raise ValueError("Waypoint times must be strictly increasing.")

            previous_time = current_waypoint.time

        self._waypoints = candidate

    def remove_waypoint(
        self,
        index: int,
    ) -> Waypoint:
        """
        Remove and return a waypoint.
        """

        return self._waypoints.pop(index)

    def clear(
        self,
    ) -> None:
        """
        Remove every waypoint.
        """

        self._waypoints.clear()

    def get_waypoint(
        self,
        index: int,
    ) -> Waypoint:
        """
        Return waypoint by index.
        """

        return self._waypoints[index]

    def first_waypoint(
        self,
    ) -> Waypoint:
        """
        Return the first waypoint.
        """

        if not self._waypoints:
            raise ValueError("Trajectory is empty.")

        return self._waypoints[0]

    def last_waypoint(
        self,
    ) -> Waypoint:
        """
        Return the final waypoint.
        """

        if not self._waypoints:
            raise ValueError("Trajectory is empty.")

        return self._waypoints[-1]

    def is_empty(
        self,
    ) -> bool:
        """
        Return True if no waypoints exist.
        """

        return len(self._waypoints) == 0

    def num_waypoints(
        self,
    ) -> int:
        """
        Number of stored waypoints.
        """

        return len(self._waypoints)

    def duration(
        self,
    ) -> float:
        """
        Mission duration.
        """

        if self.is_empty():
            return 0.0

        return self._waypoints[-1].time - self._waypoints[0].time

    def validate(
        self,
    ) -> None:
        """
        Validate the trajectory.
        """

        previous_time = -np.inf

        for waypoint in self._waypoints:

            if waypoint.time <= previous_time:
                raise ValueError("Waypoint times must be strictly increasing.")

            previous_time = waypoint.time

    def as_list(
        self,
    ) -> list[Waypoint]:
        """
        Return a deep copy of the waypoint list.
        """

        return [waypoint.copy() for waypoint in self._waypoints]

    def copy(
        self,
    ) -> "ReferenceTrajectory":
        """
        Return a deep copy of the trajectory.
        """

        return ReferenceTrajectory(self.as_list())

    def __len__(
        self,
    ) -> int:
        """
        Return the number of waypoints.
        """

        return len(self._waypoints)

    def __bool__(
        self,
    ) -> bool:
        """
        Return True if the trajectory contains waypoints.
        """

        return not self.is_empty()

    def __iter__(
        self,
    ) -> Iterator[Waypoint]:
        """
        Iterate over waypoints.
        """

        return iter(self._waypoints)

    def __getitem__(
        self,
        index: int | slice,
    ) -> Waypoint | list[Waypoint]:
        """
        Return waypoint(s) by index or slice.
        """

        return self._waypoints[index]

    def __contains__(
        self,
        waypoint: object,
    ) -> bool:
        """
        Return True if the waypoint exists.
        """

        return waypoint in self._waypoints

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation of the trajectory.
        """

        return (
            f"{self.__class__.__name__}"
            f"(num_waypoints={len(self)}, "
            f"duration={self.duration():.3f} s)"
        )

    @property
    def waypoints(
        self,
    ) -> tuple[Waypoint, ...]:
        """
        Return the trajectory waypoints as an immutable tuple.
        """

        return tuple(self._waypoints)


__all__ = [
    "Waypoint",
    "ReferenceTrajectory",
]
