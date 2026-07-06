"""
Reference trajectory builder for RocketPy GNC.

This module converts RocketPy flight simulation results into
ReferenceTrajectory objects that can be consumed by the Guidance
subsystem.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.guidance.trajectory import ReferenceTrajectory, Waypoint
from rocketpy.gnc.navigation.quaternion import Quaternion


class TrajectoryBuilder:
    """
    Reference trajectory builder.

    The builder converts RocketPy flight state
    histories into ReferenceTrajectory objects
    used by the Guidance subsystem.
    """

    def from_flight(
        self,
        flight: Any,
    ) -> ReferenceTrajectory:
        """
        Build a reference trajectory from a RocketPy Flight.

        The returned trajectory is constructed from the
        flight solution stored in the supplied Flight
        object.

        Parameters
        ----------
        flight : Any
            RocketPy Flight object.

        Returns
        -------
        ReferenceTrajectory
            Reference trajectory constructed from the
            supplied flight solution.
        """

        if not hasattr(
            flight,
            "solution",
        ):
            raise TypeError("flight must provide a 'solution' attribute.")

        return self.from_solution(flight.solution)

    def from_solution(
        self,
        solution: Any,
    ) -> ReferenceTrajectory:
        """
        Build a reference trajectory from a RocketPy solution.

        The returned trajectory is constructed by
        converting each solution state into a
        reference waypoint.


        Parameters
        ----------
        solution : Any
            RocketPy flight solution.

        Returns
        -------
        ReferenceTrajectory
            Reference trajectory constructed from the
            supplied RocketPy solution.
        """

        states = list(solution)

        self._validate_solution(states)

        trajectory = ReferenceTrajectory()

        for state in states:
            waypoint = self._build_waypoint(state)
            trajectory.add_waypoint(waypoint)

        return trajectory

    def _validate_solution(
        self,
        states: list[Any],
    ) -> None:
        """
        Validate a RocketPy flight solution.

        Ensures the supplied solution contains valid
        state vectors with strictly increasing
        timestamps.


        Parameters
        ----------
        states : list[Any]
            Flight solution states to validate.
        """

        if states is None:
            raise ValueError("solution cannot be None.")

        if len(states) == 0:
            raise ValueError("solution is empty.")

        previous_time = -np.inf

        for index, state in enumerate(states):

            state = np.asarray(
                state,
                dtype=np.float64,
            )

            if state.shape != (14,):
                raise ValueError(f"State {index} must contain exactly 14 elements.")

            if not np.all(np.isfinite(state)):
                raise ValueError(f"State {index} contains non-finite values.")

            if state[0] <= previous_time:
                raise ValueError("Solution timestamps must be strictly increasing.")

            previous_time = state[0]

    def _build_waypoint(
        self,
        state: NDArray[np.float64] | list[float],
    ) -> Waypoint:
        """
        Build a reference waypoint from a RocketPy state.

        The returned waypoint is constructed from the
        supplied RocketPy state vector.

        Parameters
        ----------
        state : array-like
            RocketPy state vector.

        Returns
        -------
        Waypoint
            Reference waypoint constructed from the
            supplied state vector.
        """

        state = np.asarray(
            state,
            dtype=np.float64,
        )

        return Waypoint(
            time=float(state[0]),
            position=np.asarray(
                state[1:4],
                dtype=np.float64,
            ),
            velocity=np.asarray(
                state[4:7],
                dtype=np.float64,
            ),
            attitude=Quaternion(
                state[7],
                state[8],
                state[9],
                state[10],
            ),
            angular_velocity=np.asarray(
                state[11:14],
                dtype=np.float64,
            ),
        )


__all__ = [
    "TrajectoryBuilder",
]
