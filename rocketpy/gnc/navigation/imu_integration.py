"""Strapdown IMU integration for RocketGNC.

This module provides a lightweight inertial navigation propagator that updates
attitude, velocity, and position estimates from body-frame gyroscope and
accelerometer measurements using the existing quaternion and frame utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.navigation.frames import body_to_ned
from rocketpy.gnc.navigation.quaternion import Quaternion


class IMUIntegrator:
    """
    Integrate strapdown IMU measurements into navigation state.

    The integrator propagates attitude in the body-to-navigation frame using the
    Hamilton quaternion convention and updates velocity and position in the
    navigation frame using first-order integration.
    """

    def __init__(
        self,
        initial_position: NDArray[np.float64] | list[float] | tuple[float, ...],
        initial_velocity: NDArray[np.float64] | list[float] | tuple[float, ...],
        initial_attitude: Quaternion,
        gravity: NDArray[np.float64] | list[float] | tuple[float, ...],
    ) -> None:
        """
        Initialize the integrator with an initial navigation state.

        Parameters
        ----------
        initial_position : array-like of shape (3,)
            Initial position in the navigation frame.
        initial_velocity : array-like of shape (3,)
            Initial velocity in the navigation frame.
        initial_attitude : Quaternion
            Initial body-to-navigation attitude quaternion.
        gravity : array-like of shape (3,)
            Gravity vector expressed in the navigation frame.
        """
        self._initial_position = self._validate_vector(
            initial_position, "initial_position"
        )
        self._initial_velocity = self._validate_vector(
            initial_velocity, "initial_velocity"
        )
        self._initial_attitude = self._validate_quaternion(
            initial_attitude, "initial_attitude"
        )
        self._initial_gravity = self._validate_vector(gravity, "gravity")

        self._position = self._initial_position.copy()
        self._velocity = self._initial_velocity.copy()
        self._attitude = self._initial_attitude.normalized().copy()
        self._gravity = self._initial_gravity.copy()

    @property
    def position(self) -> NDArray[np.float64]:
        """
        Return the current position state.

        The returned position vector is copied from the
        internally stored navigation position.
        """
        return self._position.copy()

    @property
    def velocity(self) -> NDArray[np.float64]:
        """
        Return the current velocity state.

        The returned velocity vector is copied from the
        internally stored navigation velocity.
        """
        return self._velocity.copy()

    @property
    def attitude(self) -> Quaternion:
        """
        Return the current attitude quaternion.

        The returned quaternion is copied from the
        internally stored navigation attitude.
        """
        return self._attitude.copy()

    def update(
        self,
        gyro: NDArray[np.float64] | list[float] | tuple[float, ...],
        accel: NDArray[np.float64] | list[float] | tuple[float, ...],
        dt: float | np.floating[Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], Quaternion]:
        """ "
        Propagate the navigation state by one IMU integration step.

        The updated position, velocity and attitude are
        computed from the supplied IMU gyroscope and
        accelerometer measurements together with the
        current navigation state.

        Parameters
        ----------
        gyro : array-like of shape (3,)
            Body-frame angular velocity in rad/s.
        accel : array-like of shape (3,)
            IMU specific force measured in the body frame in m/s^2.
        dt : float
            Integration step size in seconds.

        Returns
        -------
        tuple
            Updated position, velocity, and attitude state.

        Notes
        -----
        The gyroscope measurement is interpreted as body-frame angular
        velocity, while the accelerometer measurement is interpreted as IMU
        specific force in the body frame. The specific force is rotated into
        the navigation frame using the current attitude and gravity is added in
        the navigation frame to obtain inertial acceleration.
        """
        gyro_vector = self._validate_vector(gyro, "gyro")
        accel_vector = self._validate_vector(accel, "accel")
        dt_value = self._validate_dt(dt)

        self._attitude.integrate(gyro_vector, dt_value)
        self._attitude.normalize()

        nav_accel = body_to_ned(accel_vector, self._attitude)
        nav_accel = nav_accel + self._gravity

        old_velocity = self._velocity.copy()
        self._velocity = self._velocity + nav_accel * np.float64(dt_value)
        self._position = (
            self._position
            + old_velocity * np.float64(dt_value)
            + 0.5 * nav_accel * np.float64(dt_value) ** 2
        )

        return self.get_state()

    def reset(self) -> None:
        """
        Reset the integrator.

        The navigation state is restored from the
        internally stored initial position, velocity,
        attitude and gravity values.
        """
        self._position = self._initial_position.copy()
        self._velocity = self._initial_velocity.copy()
        self._attitude = self._initial_attitude.normalized().copy()
        self._gravity = self._initial_gravity.copy()

    def get_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64], Quaternion]:
        """
        Return the current navigation state.

        The returned position, velocity and attitude are
        copied from the internally stored navigation state.
        """
        return self.position, self.velocity, self.attitude

    def set_state(
        self,
        position: NDArray[np.float64] | list[float] | tuple[float, ...],
        velocity: NDArray[np.float64] | list[float] | tuple[float, ...],
        attitude: Quaternion,
    ) -> None:
        """
        Replace the navigation state.

        The supplied position, velocity and attitude are
        validated and copied into the internal state.

        Parameters
        ----------
        position : array-like of shape (3,)
            New navigation-frame position.
        velocity : array-like of shape (3,)
            New navigation-frame velocity.
        attitude : Quaternion
            New body-to-navigation attitude quaternion.
        """
        self._position = self._validate_vector(position, "position")
        self._velocity = self._validate_vector(velocity, "velocity")
        self._attitude = (
            self._validate_quaternion(attitude, "attitude").normalized().copy()
        )

    def _validate_vector(
        self,
        vector: NDArray[np.float64] | list[float] | tuple[float, ...],
        name: str,
    ) -> NDArray[np.float64]:
        """
        Validate a three-dimensional vector.

        Returns the validated vector converted to
        float64 for IMU integration computations.
        """
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != 3:
            raise ValueError(f"{name} must contain exactly three components.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        return np.asarray(values, dtype=np.float64)

    def _validate_quaternion(self, quaternion: Any, name: str) -> Quaternion:
        """
        Validate a quaternion.

        Returns the validated quaternion instance
        unchanged for subsequent computations.

        """
        if not isinstance(quaternion, Quaternion):
            raise TypeError(f"{name} must be a Quaternion instance.")
        return quaternion

    def _validate_dt(self, dt: float | np.floating[Any]) -> np.float64:
        """
        Validate the integration time step.

        Returns the validated timestep as float64.
        """
        dt_value = np.float64(dt)
        if not np.isfinite(dt_value):
            raise ValueError("dt must be finite.")
        if dt_value <= 0.0:
            raise ValueError("dt must be greater than zero.")
        return dt_value


__all__ = ["IMUIntegrator"]
