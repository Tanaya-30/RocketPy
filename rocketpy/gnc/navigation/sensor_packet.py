"""
Sensor packet representation for RocketPy GNC Navigation.

This module defines the SensorPacket class, which represents a single
time-stamped set of measurements received from onboard sensors.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorPacket:
    """
    Immutable sensor measurement packet.

    Parameters
    ----------
    timestamp : float
        Measurement timestamp in seconds.

    accelerometer : numpy.ndarray
        3-axis accelerometer measurement [ax, ay, az] in m/s².

    gyroscope : numpy.ndarray
        3-axis gyroscope measurement [gx, gy, gz] in rad/s.

    gps_position : numpy.ndarray
        GPS position [latitude, longitude, altitude].

    barometric_altitude : float
        Altitude estimated from the barometric sensor in meters.
    """

    timestamp: float
    accelerometer: np.ndarray
    gyroscope: np.ndarray
    gps_position: np.ndarray
    barometric_altitude: float

    def __post_init__(self):
        """Validate sensor packet contents."""

        if self.timestamp < 0:
            raise ValueError("Timestamp must be non-negative.")

        self._validate_vector(
            self.accelerometer,
            "accelerometer",
        )

        self._validate_vector(
            self.gyroscope,
            "gyroscope",
        )

        self._validate_vector(
            self.gps_position,
            "gps_position",
        )

        if not np.isfinite(self.barometric_altitude):
            raise ValueError(
                "Barometric altitude must be finite."
            )

    @staticmethod
    def _validate_vector(
        vector: np.ndarray,
        name: str,
    ):
        """
        Validate a 3-element sensor vector.

        Parameters
        ----------
        vector : numpy.ndarray
            Sensor vector.

        name : str
            Name of the sensor.
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError(
                f"{name} must be a NumPy array."
            )

        if vector.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,), got {vector.shape}."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} contains non-finite values."
            )