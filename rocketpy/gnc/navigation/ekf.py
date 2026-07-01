# """Lightweight extended Kalman filter for RocketGNC navigation.

# This module implements a compact state estimator for rocket navigation that
# propagates a 13-state vector with IMU measurements and corrects it with GNSS
# and barometer observations. The implementation is intentionally lightweight and
# keeps the state representation compatible with the existing quaternion and IMU
# integration utilities.
# """

# from __future__ import annotations

# from typing import Any

# import numpy as np
# from numpy.typing import NDArray

# from rocketpy.gnc.navigation.frames import validate_vector
# from rocketpy.gnc.navigation.imu_integration import IMUIntegrator
# from rocketpy.gnc.navigation.quaternion import Quaternion
# from rocketpy.gnc.navigation.sensor_fusion import SensorFusion


# class ExtendedKalmanFilter:
#     """Perform a lightweight EKF prediction and correction cycle.

#     The filter state is a 13-element vector containing position, velocity,
#     attitude quaternion, and angular velocity:

#     ``[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]``

#     The implementation uses the existing quaternion and IMU integration modules
#     for state propagation and keeps the covariance matrix symmetric and
#     numerically stable.
#     """

#     def __init__(self) -> None:
#         """Initialize the filter with a default state and covariance."""
#         self._state: NDArray[np.float64] = self._initial_state()
#         self._covariance: NDArray[np.float64] = self._initial_covariance()
#         self._process_noise: NDArray[np.float64] = np.eye(13, dtype=np.float64) * np.float64(1e-3)
#         self._gnss_measurement_noise: NDArray[np.float64] = np.eye(3, dtype=np.float64) * np.float64(1e-2)
#         self._barometer_measurement_noise: NDArray[np.float64] = np.array([[1e-2]], dtype=np.float64)
#         self._gravity: NDArray[np.float64] = np.array([0.0, 0.0, -9.81], dtype=np.float64)
#         self._sensor_fusion: SensorFusion = SensorFusion()

#     def predict(
#         self,
#         gyro: NDArray[np.float64] | list[float] | tuple[float, ...],
#         accel: NDArray[np.float64] | list[float] | tuple[float, ...],
#         dt: float | np.floating[Any],
#     ) -> NDArray[np.float64]:
#         """Propagate the state and covariance with an IMU prediction step.

#         Parameters
#         ----------
#         gyro : array-like of shape (3,)
#             Body-frame gyroscope measurement in rad/s.
#         accel : array-like of shape (3,)
#             Body-frame accelerometer measurement in m/s^2.
#         dt : float
#             Prediction time step in seconds.

#         Returns
#         -------
#         numpy.ndarray
#             The predicted state vector as a float64 array of shape (13,).
#         """
#         gyro_vector = self._validate_vector(gyro, "gyro")
#         accel_vector = self._validate_vector(accel, "accel")
#         dt_value = self._validate_dt(dt)

#         quaternion = Quaternion.from_numpy(self._state[6:10]).normalized()
#         integrator = IMUIntegrator(
#             self._state[0:3],
#             self._state[3:6],
#             quaternion,
#             self._gravity,
#         )
#         position, velocity, attitude = integrator.update(gyro_vector, accel_vector, dt_value)

#         predicted_state = np.empty(13, dtype=np.float64)
#         predicted_state[0:3] = np.asarray(position, dtype=np.float64)
#         predicted_state[3:6] = np.asarray(velocity, dtype=np.float64)
#         predicted_state[6:10] = np.asarray(attitude.normalized().as_numpy(), dtype=np.float64)
#         predicted_state[10:13] = np.asarray(gyro_vector, dtype=np.float64)

#         transition = self._state_transition_matrix(dt_value)
#         covariance = transition @ self._covariance @ transition.T + self._process_noise
#         covariance = self._symmetrize_covariance(covariance)

#         self._state = predicted_state
#         self._covariance = covariance
#         return self._state.copy()

#     def update_gnss(
#         self,
#         position: NDArray[np.float64] | list[float] | tuple[float, ...],
#     ) -> NDArray[np.float64]:
#         """Apply a GNSS position update using a standard EKF correction.

#         Parameters
#         ----------
#         position : array-like of shape (3,)
#             Position measurement in the navigation frame.

#         Returns
#         -------
#         numpy.ndarray
#             The updated state vector as a float64 array of shape (13,).
#         """
#         measurement = self._validate_vector(position, "position")
#         measurement_model = self._position_measurement_matrix()
#         innovation = measurement - measurement_model @ self._state
#         covariance = measurement_model @ self._covariance @ measurement_model.T + self._gnss_measurement_noise
#         covariance = self._symmetrize_covariance(covariance)

#         kalman_gain = self._kalman_gain(covariance, measurement_model)
#         self._state = self._state + kalman_gain @ innovation
#         self._state[6:10] = Quaternion.from_numpy(self._state[6:10]).normalized().as_numpy()

#         identity = np.eye(13, dtype=np.float64)
#         self._covariance = (
#             identity - kalman_gain @ measurement_model
#         ) @ self._covariance @ (
#             identity - kalman_gain @ measurement_model
#         ).T + kalman_gain @ self._gnss_measurement_noise @ kalman_gain.T
#         self._covariance = self._symmetrize_covariance(self._covariance)
#         return self._state.copy()

#     def update_barometer(self, altitude: float | np.floating[Any]) -> NDArray[np.float64]:
#         """Apply a barometer altitude update using a scalar EKF correction.

#         Parameters
#         ----------
#         altitude : float
#             Altitude measurement in metres.

#         Returns
#         -------
#         numpy.ndarray
#             The updated state vector as a float64 array of shape (13,).
#         """
#         altitude_value = self._validate_scalar(altitude, "altitude")
#         measurement_model = self._altitude_measurement_matrix()
#         innovation = np.array([altitude_value], dtype=np.float64) - measurement_model @ self._state
#         covariance = measurement_model @ self._covariance @ measurement_model.T + self._barometer_measurement_noise
#         covariance = self._symmetrize_covariance(covariance)

#         kalman_gain = self._kalman_gain(covariance, measurement_model)
#         self._state = self._state + kalman_gain @ innovation
#         self._state[6:10] = Quaternion.from_numpy(self._state[6:10]).normalized().as_numpy()

#         identity = np.eye(13, dtype=np.float64)
#         self._covariance = (
#             identity - kalman_gain @ measurement_model
#         ) @ self._covariance @ (
#             identity - kalman_gain @ measurement_model
#         ).T + kalman_gain @ self._barometer_measurement_noise @ kalman_gain.T
#         self._covariance = self._symmetrize_covariance(self._covariance)
#         return self._state.copy()

#     def get_state(self) -> NDArray[np.float64]:
#         """Return a copy of the current EKF state vector."""
#         return self._state.copy()

#     def set_state(self, state: NDArray[np.float64] | list[float] | tuple[float, ...]) -> None:
#         """Set the filter state vector.

#         Parameters
#         ----------
#         state : array-like of shape (13,)
#             New state vector in the form ``[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]``.
#         """
#         values = np.asarray(state, dtype=np.float64).reshape(-1)
#         if values.size != 13:
#             raise ValueError("State must contain exactly thirteen components.")
#         if not np.all(np.isfinite(values)):
#             raise ValueError("State must contain only finite values.")
#         self._state = values.astype(np.float64, copy=False)
#         self._state[6:10] = Quaternion.from_numpy(self._state[6:10]).normalized().as_numpy()

#     def reset(self) -> None:
#         """Reset the filter state and covariance to their defaults."""
#         self._state = self._initial_state()
#         self._covariance = self._initial_covariance()

#     def _initial_state(self) -> NDArray[np.float64]:
#         """Return the default initial state vector."""
#         return np.array(
#             [
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 1.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#             ],
#             dtype=np.float64,
#         )

#     def _initial_covariance(self) -> NDArray[np.float64]:
#         """Return the default initial covariance matrix."""
#         covariance = np.eye(13, dtype=np.float64)
#         covariance[6:10, 6:10] = np.eye(4, dtype=np.float64) * np.float64(1e-4)
#         covariance[10:13, 10:13] = np.eye(3, dtype=np.float64) * np.float64(1e-2)
#         return covariance.astype(np.float64, copy=False)

#     def _validate_vector(
#         self,
#         vector: NDArray[np.float64] | list[float] | tuple[float, ...],
#         name: str,
#     ) -> NDArray[np.float64]:
#         """Validate a three-component vector and return it as float64."""
#         values = validate_vector(vector)
#         if not np.all(np.isfinite(values)):
#             raise ValueError(f"{name} must contain only finite values.")
#         return values.astype(np.float64, copy=False)

#     def _validate_scalar(self, value: float | np.floating[Any], name: str) -> np.float64:
#         """Validate a scalar value and return it as float64."""
#         scalar = np.float64(value)
#         if not np.isfinite(scalar):
#             raise ValueError(f"{name} must be finite.")
#         return scalar

#     def _validate_dt(self, dt: float | np.floating[Any]) -> np.float64:
#         """Validate a prediction time step and return it as float64."""
#         dt_value = np.float64(dt)
#         if not np.isfinite(dt_value):
#             raise ValueError("dt must be finite.")
#         if dt_value <= 0.0:
#             raise ValueError("dt must be greater than zero.")
#         return dt_value

#     def _state_transition_matrix(self, dt: np.float64) -> NDArray[np.float64]:
#         """Construct a first-order linearized state transition matrix."""
#         transition = np.eye(13, dtype=np.float64)
#         transition[0:3, 3:6] = np.eye(3, dtype=np.float64) * np.float64(dt)
#         quaternion_update = np.array(
#             [
#                 [0.0, 0.0, 0.0],
#                 [0.5 * dt, 0.0, 0.0],
#                 [0.0, 0.5 * dt, 0.0],
#                 [0.0, 0.0, 0.5 * dt],
#             ],
#             dtype=np.float64,
#         )
#         transition[6:10, 10:13] = quaternion_update
#         return transition.astype(np.float64, copy=False)

#     def _position_measurement_matrix(self) -> NDArray[np.float64]:
#         """Construct the linear position observation model."""
#         measurement_matrix = np.zeros((3, 13), dtype=np.float64)
#         measurement_matrix[0, 0] = 1.0
#         measurement_matrix[1, 1] = 1.0
#         measurement_matrix[2, 2] = 1.0
#         return measurement_matrix.astype(np.float64, copy=False)

#     def _altitude_measurement_matrix(self) -> NDArray[np.float64]:
#         """Construct the linear altitude observation model."""
#         measurement_matrix = np.zeros((1, 13), dtype=np.float64)
#         measurement_matrix[0, 2] = -1.0
#         return measurement_matrix.astype(np.float64, copy=False)

#     def _kalman_gain(
#         self,
#         innovation_covariance: NDArray[np.float64],
#         measurement_matrix: NDArray[np.float64],
#     ) -> NDArray[np.float64]:
#         """Compute a numerically stable Kalman gain."""
#         covariance_term = measurement_matrix @ self._covariance @ measurement_matrix.T
#         covariance_term = self._symmetrize_covariance(covariance_term)
#         innovation_covariance = self._symmetrize_covariance(innovation_covariance)
#         gain = self._covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
#         return gain.astype(np.float64, copy=False)

#     def _symmetrize_covariance(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
#         """Project a covariance matrix onto the symmetric positive semidefinite cone."""
#         symmetric = 0.5 * (covariance + covariance.T)
#         eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
#         eigenvalues = np.clip(eigenvalues, 0.0, None)
#         corrected = (eigenvectors * eigenvalues) @ eigenvectors.T
#         corrected = 0.5 * (corrected + corrected.T)
#         return corrected.astype(np.float64, copy=False)


# __all__ = ["ExtendedKalmanFilter"]
"""Lightweight extended Kalman filter for RocketGNC navigation.

This module implements a compact extended Kalman filter for rocket navigation.
The filter propagates a 13-state navigation vector using IMU measurements and
corrects it using GNSS and barometric altitude observations.

State vector
------------
[x, y, z,
 vx, vy, vz,
 qw, qx, qy, qz,
 wx, wy, wz]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.navigation.frames import validate_vector
from rocketpy.gnc.navigation.imu_integration import IMUIntegrator
from rocketpy.gnc.navigation.quaternion import Quaternion


class ExtendedKalmanFilter:
    """Lightweight navigation Extended Kalman Filter."""

    STATE_SIZE = 13
    POSITION_DIM = 3
    QUATERNION_DIM = 4
    ANGULAR_RATE_DIM = 3

    def __init__(self) -> None:
        """Initialize the EKF."""

        self._gravity = np.array(
            [0.0, 0.0, -9.81],
            dtype=np.float64,
        )

        self._state = self._initial_state()
        self._covariance = self._initial_covariance()

        self._process_noise = (
            np.eye(self.STATE_SIZE, dtype=np.float64)
            * np.float64(1e-3)
        )

        self._gnss_measurement_noise = (
            np.eye(3, dtype=np.float64)
            * np.float64(1e-2)
        )

        self._barometer_measurement_noise = np.array(
            [[1e-2]],
            dtype=np.float64,
        )
    
    def predict(
        self,
        gyro: NDArray[np.float64] | list[float] | tuple[float, ...],
        accel: NDArray[np.float64] | list[float] | tuple[float, ...],
        dt: float | np.floating[Any],
    ) -> NDArray[np.float64]:
        """Propagate the EKF state using IMU measurements.

        Parameters
        ----------
        gyro : array-like of shape (3,)
            Body-frame angular velocity in rad/s.
        accel : array-like of shape (3,)
            Body-frame specific force in m/s².
        dt : float
            Integration time step in seconds.

        Returns
        -------
        numpy.ndarray
            Updated 13-state navigation vector.
        """
        gyro_vector = self._validate_vector(gyro, "gyro")
        accel_vector = self._validate_vector(accel, "accel")
        dt_value = self._validate_dt(dt)

        quaternion = Quaternion.from_numpy(
            self._state[6:10]
        ).normalized()

        integrator = IMUIntegrator(
            initial_position=self._state[0:3],
            initial_velocity=self._state[3:6],
            initial_attitude=quaternion,
            gravity=self._gravity,
        )

        position, velocity, attitude = integrator.update(
            gyro=gyro_vector,
            accel=accel_vector,
            dt=dt_value,
        )

        predicted_state = np.empty_like(self._state)

        predicted_state[0:3] = position
        predicted_state[3:6] = velocity
        predicted_state[6:10] = attitude.normalized().as_numpy()
        predicted_state[10:13] = gyro_vector

        transition = self._state_transition_matrix(dt_value)

        self._covariance = (
            transition
            @ self._covariance
            @ transition.T
            + self._process_noise
        )

        self._covariance = self._symmetrize_covariance(
            self._covariance
        )

        self._state = predicted_state

        return self.get_state()
    
    def update_gnss(
        self,
        position: NDArray[np.float64] | list[float] | tuple[float, ...],
    ) -> NDArray[np.float64]:
        """Correct the navigation state using a GNSS position measurement.

        Parameters
        ----------
        position : array-like of shape (3,)
            Position measurement expressed in the navigation frame.

        Returns
        -------
        numpy.ndarray
            Corrected state vector.
        """
        measurement = self._validate_vector(position, "position")

        measurement_matrix = self._position_measurement_matrix()

        innovation = (
            measurement
            - measurement_matrix @ self._state
        )

        innovation_covariance = (
            measurement_matrix
            @ self._covariance
            @ measurement_matrix.T
            + self._gnss_measurement_noise
        )

        innovation_covariance = self._symmetrize_covariance(
            innovation_covariance
        )

        kalman_gain = self._kalman_gain(
            innovation_covariance,
            measurement_matrix,
        )

        self._state += kalman_gain @ innovation

        self._state[6:10] = (
            Quaternion.from_numpy(
                self._state[6:10]
            )
            .normalized()
            .as_numpy()
        )

        identity = np.eye(
            self.STATE_SIZE,
            dtype=np.float64,
        )

        joseph = identity - kalman_gain @ measurement_matrix

        self._covariance = (
            joseph
            @ self._covariance
            @ joseph.T
            + kalman_gain
            @ self._gnss_measurement_noise
            @ kalman_gain.T
        )

        self._covariance = self._symmetrize_covariance(
            self._covariance
        )

        return self.get_state()
    
    def update_barometer(
        self,
        altitude: float | np.floating[Any],
    ) -> NDArray[np.float64]:
        """Correct the navigation state using a barometer altitude measurement."""
        altitude_value = self._validate_scalar(
            altitude,
            "altitude",
        )

        measurement_matrix = self._altitude_measurement_matrix()

        innovation = (
            np.array([altitude_value], dtype=np.float64)
            - measurement_matrix @ self._state
        )

        innovation_covariance = (
            measurement_matrix
            @ self._covariance
            @ measurement_matrix.T
            + self._barometer_measurement_noise
        )

        innovation_covariance = self._symmetrize_covariance(
            innovation_covariance
        )

        kalman_gain = self._kalman_gain(
            innovation_covariance,
            measurement_matrix,
        )

        self._state += kalman_gain @ innovation

        self._state[6:10] = (
            Quaternion.from_numpy(
                self._state[6:10]
            )
            .normalized()
            .as_numpy()
        )

        identity = np.eye(
            self.STATE_SIZE,
            dtype=np.float64,
        )

        joseph = identity - kalman_gain @ measurement_matrix

        self._covariance = (
            joseph
            @ self._covariance
            @ joseph.T
            + kalman_gain
            @ self._barometer_measurement_noise
            @ kalman_gain.T
        )

        self._covariance = self._symmetrize_covariance(
            self._covariance
        )

        return self.get_state()

    def get_state(self) -> NDArray[np.float64]:
        """Return a copy of the current state."""
        return self._state.copy()

    def set_state(
        self,
        state: NDArray[np.float64]
        | list[float]
        | tuple[float, ...],
    ) -> None:
        """Replace the EKF state vector."""
        values = np.asarray(
            state,
            dtype=np.float64,
        ).reshape(-1)

        if values.size != self.STATE_SIZE:
            raise ValueError(
                f"State must contain exactly {self.STATE_SIZE} values."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "State must contain only finite values."
            )

        values[6:10] = (
            Quaternion.from_numpy(values[6:10])
            .normalized()
            .as_numpy()
        )

        self._state = values.copy()

    def get_covariance(self) -> NDArray[np.float64]:
        """Return a copy of the covariance matrix."""
        return self._covariance.copy()

    def set_covariance(
        self,
        covariance: NDArray[np.float64]
        | list[list[float]],
    ) -> None:
        """Replace the covariance matrix."""
        values = np.asarray(
        covariance,
        dtype=np.float64,
    )

        if values.shape != (
            self.STATE_SIZE,
            self.STATE_SIZE,
        ):
            raise ValueError(
                f"Covariance must have shape ({self.STATE_SIZE}, {self.STATE_SIZE})."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Covariance must contain only finite values."
            )

        if not np.allclose(values, values.T, atol=1e-12):
            raise ValueError(
                "Covariance matrix must be symmetric."
            )

        if np.any(np.diag(values) < 0.0):
            raise ValueError(
                "Covariance diagonal entries must be non-negative."
            )

        self._covariance = values.copy()

    def reset(self) -> None:
        """Reset the filter to its default state."""
        self._state = self._initial_state()
        self._covariance = self._initial_covariance()

        self._process_noise = (
            np.eye(
                self.STATE_SIZE,
                dtype=np.float64,
            )
            * np.float64(1e-3)
        )

        self._gnss_measurement_noise = (
            np.eye(
                3,
                dtype=np.float64,
            )
            * np.float64(1e-2)
        )

        self._barometer_measurement_noise = np.array(
            [[1e-2]],
            dtype=np.float64,
        )

    def _initial_state(self) -> NDArray[np.float64]:
        """Return the default EKF state."""
        state = np.zeros(
            self.STATE_SIZE,
            dtype=np.float64,
        )
        state[6] = 1.0
        return state

    def _initial_covariance(self) -> NDArray[np.float64]:
        """Return the default covariance matrix."""
        covariance = np.eye(
            self.STATE_SIZE,
            dtype=np.float64,
        )

        covariance[6:10, 6:10] *= np.float64(1e-4)
        covariance[10:13, 10:13] *= np.float64(1e-2)

        return covariance

    def _validate_vector(
        self,
        vector: NDArray[np.float64]
        | list[float]
        | tuple[float, ...],
        name: str,
    ) -> NDArray[np.float64]:
        """Validate a three-component vector."""
        values = validate_vector(vector)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} must contain only finite values."
            )

        return values.astype(np.float64, copy=False)

    def _validate_scalar(
        self,
        value: float | np.floating[Any],
        name: str,
    ) -> np.float64:
        """Validate a scalar value."""
        scalar = np.float64(value)

        if not np.isfinite(scalar):
            raise ValueError(
                f"{name} must be finite."
            )

        return scalar

    def _validate_dt(
        self,
        dt: float | np.floating[Any],
    ) -> np.float64:
        """Validate the prediction time step."""
        dt_value = np.float64(dt)

        if not np.isfinite(dt_value):
            raise ValueError("dt must be finite.")

        if dt_value <= 0.0:
            raise ValueError(
                "dt must be greater than zero."
            )

        return dt_value

    def _state_transition_matrix(
        self,
        dt: np.float64,
    ) -> NDArray[np.float64]:
        """Construct a first-order state transition matrix."""
        transition = np.eye(
            self.STATE_SIZE,
            dtype=np.float64,
        )

        transition[0:3, 3:6] = (
            np.eye(3, dtype=np.float64)
            * dt
        )

        transition[6:10, 10:13] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5 * dt, 0.0, 0.0],
                [0.0, 0.5 * dt, 0.0],
                [0.0, 0.0, 0.5 * dt],
            ],
            dtype=np.float64,
        )

        return transition

    def _position_measurement_matrix(
        self,
    ) -> NDArray[np.float64]:
        """Return the GNSS observation matrix."""
        measurement = np.zeros(
            (3, self.STATE_SIZE),
            dtype=np.float64,
        )

        measurement[0, 0] = 1.0
        measurement[1, 1] = 1.0
        measurement[2, 2] = 1.0

        return measurement

    def _altitude_measurement_matrix(
        self,
    ) -> NDArray[np.float64]:
        """Return the barometer observation matrix."""
        measurement = np.zeros(
            (1, self.STATE_SIZE),
            dtype=np.float64,
        )

        measurement[0, 2] = -1.0

        return measurement

    def _kalman_gain(
        self,
        innovation_covariance: NDArray[np.float64],
        measurement_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the Kalman gain."""
        innovation_covariance = (
            self._symmetrize_covariance(
                innovation_covariance
            )
        )

        gain = (
            self._covariance
            @ measurement_matrix.T
            @ np.linalg.inv(
                innovation_covariance
            )
        )

        return gain.astype(
            np.float64,
            copy=False,
        )

    def _symmetrize_covariance(
        self,
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Force a covariance matrix to remain symmetric."""
        symmetric = (
            covariance + covariance.T
        ) * np.float64(0.5)

        eigenvalues, eigenvectors = np.linalg.eigh(
            symmetric
        )

        eigenvalues = np.clip(
            eigenvalues,
            0.0,
            None,
        )

        corrected = (
            eigenvectors
            @ np.diag(eigenvalues)
            @ eigenvectors.T
        )

        return (
            (
                corrected + corrected.T
            )
            * np.float64(0.5)
        ).astype(
            np.float64,
            copy=False,
        )


__all__ = [
    "ExtendedKalmanFilter",
]