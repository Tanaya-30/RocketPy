"""Lightweight sensor fusion measurement collection for RocketGNC.

This module collects and validates the latest IMU, GNSS, and barometer
measurements before they are consumed by downstream navigation and estimation
components. The implementation is intentionally limited to measurement
validation, storage, and accessors; it does not estimate bias, integrate IMU
measurements, or implement any Kalman filtering logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.interfaces import BaroPacket, GNSSPacket, IMUPacket
from rocketpy.gnc.navigation.frames import validate_vector


class SensorFusion:
    """Collect and validate the latest sensor measurements.

    The class maintains the most recent validated reading from each supported
    sensor and exposes it through a small access interface. Timestamps must be
    finite, non-negative, and strictly increasing for each sensor stream.
    """

    def __init__(self) -> None:
        """Initialize the fusion container with empty measurements."""
        self._latest_imu: IMUPacket | None = None
        self._latest_gnss: GNSSPacket | None = None
        self._latest_barometer: BaroPacket | None = None

        self._imu_timestamp: np.float64 | None = None
        self._gnss_timestamp: np.float64 | None = None
        self._barometer_timestamp: np.float64 | None = None

    def update_imu(
        self,
        gyro: NDArray[np.float64] | list[float] | tuple[float, ...],
        accel: NDArray[np.float64] | list[float] | tuple[float, ...],
        timestamp: float | np.floating[Any],
    ) -> None:
        """Update the latest IMU measurement.

        Parameters
        ----------
        gyro : array-like of shape (3,)
            Gyroscope measurement in rad/s expressed in the body frame.
        accel : array-like of shape (3,)
            Accelerometer measurement in m/s^2 expressed in the body frame.
        timestamp : float
            Measurement timestamp in seconds.

        Raises
        ------
        ValueError
            If the timestamp is negative, non-finite, or not strictly greater
            than the previous IMU timestamp.
        """
        gyro_vector = self._validate_vector(gyro, "gyro")
        accel_vector = self._validate_vector(accel, "accel")
        timestamp_value = self._validate_timestamp(timestamp, "timestamp")

        self._validate_monotonic_timestamp(timestamp_value, "imu")

        self._latest_imu = IMUPacket(
            time=float(timestamp_value),
            accel_x=float(np.float64(accel_vector[0])),
            accel_y=float(np.float64(accel_vector[1])),
            accel_z=float(np.float64(accel_vector[2])),
            gyro_x=float(np.float64(gyro_vector[0])),
            gyro_y=float(np.float64(gyro_vector[1])),
            gyro_z=float(np.float64(gyro_vector[2])),
        )
        self._imu_timestamp = timestamp_value

    def update_gnss(
        self,
        latitude: float | np.floating[Any],
        longitude: float | np.floating[Any],
        altitude: float | np.floating[Any],
        timestamp: float | np.floating[Any],
    ) -> None:
        """Update the latest GNSS measurement.

        Parameters
        ----------
        latitude : float
            Geodetic latitude in degrees.
        longitude : float
            Geodetic longitude in degrees.
        altitude : float
            Altitude above mean sea level in metres.
        timestamp : float
            Measurement timestamp in seconds.
        """
        latitude_value = self._validate_scalar(latitude, "latitude")
        longitude_value = self._validate_scalar(longitude, "longitude")
        altitude_value = self._validate_scalar(altitude, "altitude")
        timestamp_value = self._validate_timestamp(timestamp, "timestamp")

        if not (-90.0 <= latitude_value <= 90.0):
            raise ValueError("Latitude must be between -90 and 90 degrees.")

        if not (-180.0 <= longitude_value <= 180.0):
            raise ValueError("Longitude must be between -180 and 180 degrees.")

        self._validate_monotonic_timestamp(timestamp_value, "gnss")

        self._latest_gnss = GNSSPacket(
            time=float(timestamp_value),
            latitude=float(latitude_value),
            longitude=float(longitude_value),
            altitude_m=float(altitude_value),
            valid=True,
        )
        self._gnss_timestamp = timestamp_value

    def update_barometer(
        self,
        pressure: float | np.floating[Any],
        timestamp: float | np.floating[Any],
    ) -> None:
        """Update the latest barometer measurement.

        Parameters
        ----------
        pressure : float
            Atmospheric pressure in pascals.
        timestamp : float
            Measurement timestamp in seconds.
        """
        pressure_value = self._validate_scalar(pressure, "pressure")
        timestamp_value = self._validate_timestamp(timestamp, "timestamp")

        if pressure_value <= 0.0:
            raise ValueError("Pressure must be positive.")

        self._validate_monotonic_timestamp(timestamp_value, "barometer")

        self._latest_barometer = BaroPacket(
            time=float(timestamp_value),
            pressure_pa=float(pressure_value),
        )
        self._barometer_timestamp = timestamp_value

    def get_latest_measurements(self) -> dict[str, IMUPacket | GNSSPacket | BaroPacket | None]:
        """Return the latest validated measurements for all sensors."""
        return {
            "imu": self._latest_imu,
            "gnss": self._latest_gnss,
            "barometer": self._latest_barometer,
        }

    def clear(self) -> None:
        """Clear all stored measurements and timestamps."""
        self._latest_imu = None
        self._latest_gnss = None
        self._latest_barometer = None

        self._imu_timestamp = None
        self._gnss_timestamp = None
        self._barometer_timestamp = None

    def _validate_vector(
        self,
        vector: NDArray[np.float64] | list[float] | tuple[float, ...],
        name: str,
    ) -> NDArray[np.float64]:
        """Validate a 3-component vector and return it as float64."""
        values = validate_vector(vector)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        return values.astype(np.float64, copy=False)

    def _validate_scalar(self, value: float | np.floating[Any], name: str) -> np.float64:
        """Validate a scalar measurement and return it as float64."""
        scalar = np.float64(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        return scalar

    def _validate_timestamp(self, timestamp: float | np.floating[Any], name: str) -> np.float64:
        """Validate a timestamp and return it as float64."""
        value = np.float64(timestamp)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return value

    def _validate_monotonic_timestamp(self, timestamp: np.float64, sensor: str) -> None:
        """Ensure timestamps are strictly increasing for an individual sensor."""
        if sensor == "imu" and self._imu_timestamp is not None and timestamp <= self._imu_timestamp:
            raise ValueError("IMU timestamps must be strictly increasing.")
        if sensor == "gnss" and self._gnss_timestamp is not None and timestamp <= self._gnss_timestamp:
            raise ValueError("GNSS timestamps must be strictly increasing.")
        if sensor == "barometer" and self._barometer_timestamp is not None and timestamp <= self._barometer_timestamp:
            raise ValueError("Barometer timestamps must be strictly increasing.")


__all__ = ["SensorFusion"]
