"""
Reference trajectory tracking for RocketPy GNC.

This module interpolates a ReferenceTrajectory and provides the
desired waypoint corresponding to the current mission time.
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

from rocketpy.gnc.guidance.trajectory import (
    ReferenceTrajectory,
    Waypoint,
)
from rocketpy.gnc.navigation.quaternion import Quaternion


class TrajectoryTracker:
    """
    Track a reference trajectory.

    A trajectory tracker interpolates between stored waypoints to
    produce a smooth reference state.
    """

    def __init__(
        self,
        trajectory: ReferenceTrajectory | None = None,
    ) -> None:
        """
        Initialize the tracker.

        Parameters
        ----------
        trajectory : ReferenceTrajectory, optional
            Initial reference trajectory.
        """

        self._trajectory = trajectory

    @property
    def trajectory(
        self,
    ) -> ReferenceTrajectory | None:
        """
        Return the active trajectory.
        """

        return self._trajectory

    def set_trajectory(
        self,
        trajectory: ReferenceTrajectory,
    ) -> None:
        """
        Set the active trajectory.
        """

        if not isinstance(
            trajectory,
            ReferenceTrajectory,
        ):
            raise TypeError(
                "trajectory must be a ReferenceTrajectory."
            )

        self._trajectory = trajectory

    def clear(
        self,
    ) -> None:
        """
        Remove the current trajectory.
        """

        self._trajectory = None

    def has_trajectory(
        self,
    ) -> bool:
        """
        Return True if a trajectory exists.
        """

        return self._trajectory is not None
    
    def reference_at_time(
        self,
        time: float,
    ) -> Waypoint:
        """
        Return the interpolated reference waypoint at the
        requested mission time.
        """

        if self._trajectory is None:
            raise ValueError(
                "No reference trajectory available."
            )

        if self._trajectory.is_empty():
            raise ValueError(
                "Reference trajectory is empty."
            )

        first = self._trajectory.first_waypoint()
        last = self._trajectory.last_waypoint()

        if time <= first.time:
            return first.copy()

        if time >= last.time:
            return last.copy()

        index = self._find_segment(
            time,
        )

        return self._interpolate_waypoint(
            self._trajectory[index],
            self._trajectory[index + 1],
            time,
        )

    def _find_segment(
        self,
        time: float,
    ) -> int:
        """
        Locate the trajectory segment containing the
        requested time.
        """

        assert self._trajectory is not None

        waypoint_times = [
            waypoint.time
            for waypoint in self._trajectory
        ]

        index = (
            bisect_right(
                waypoint_times,
                time,
            )
            - 1
        )

        return max(
            0,
            min(
                index,
                len(self._trajectory) - 2,
            ),
        )

    @staticmethod
    def _interpolation_factor(
        first: Waypoint,
        second: Waypoint,
        time: float,
    ) -> float:
        """
        Compute the interpolation factor.
        """

        return (
            (time - first.time)
            / (second.time - first.time)
        )
    
    def _interpolate_waypoint(
        self,
        first: Waypoint,
        second: Waypoint,
        time: float,
    ) -> Waypoint:
        """
        Interpolate between two trajectory waypoints.
        """

        alpha = self._interpolation_factor(
            first,
            second,
            time,
        )

        position = (
            (1.0 - alpha) * first.position
            + alpha * second.position
        )

        velocity = (
            (1.0 - alpha) * first.velocity
            + alpha * second.velocity
        )

        angular_velocity = (
            (1.0 - alpha)
            * first.angular_velocity
            + alpha
            * second.angular_velocity
        )

        attitude = self._interpolate_quaternion(
            first.attitude,
            second.attitude,
            alpha,
        )

        return Waypoint(
            time=time,
            position=position.astype(
                np.float64,
            ),
            velocity=velocity.astype(
                np.float64,
            ),
            attitude=attitude,
            angular_velocity=angular_velocity.astype(
                np.float64,
            ),
        )

    @staticmethod
    def _interpolate_quaternion(
        first: Quaternion,
        second: Quaternion,
        alpha: float,
    ) -> Quaternion:
        """
        Interpolate between two quaternions.

        NOTE
        ----
        This implementation performs normalized linear interpolation
        (NLERP). Once the Quaternion class exposes a production SLERP
        implementation, this method can simply delegate to it without
        changing the public API.
        """

        q1 = first.as_numpy()
        q2 = second.as_numpy()

        if np.dot(q1, q2) < 0.0:
            q2 = -q2

        q = (1.0 - alpha) * q1 + alpha * q2

        q /= np.linalg.norm(q)

        return Quaternion(
            q[0],
            q[1],
            q[2],
            q[3],
        )

    def reset(
        self,
    ) -> None:
        """
        Reset the tracker.
        """

        self._trajectory = None

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        if self._trajectory is None:
            return (
                f"{self.__class__.__name__}"
                "(trajectory=None)"
            )

        return (
            f"{self.__class__.__name__}"
            f"(num_waypoints="
            f"{len(self._trajectory)})"
        )
    
__all__ = [
    "TrajectoryTracker",
]
