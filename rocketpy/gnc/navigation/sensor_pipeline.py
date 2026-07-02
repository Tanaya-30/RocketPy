"""
Navigation sensor processing pipeline for RocketPy GNC.

This module provides a high-level interface that accepts SensorPacket
measurements and propagates the navigation solution using the IMU
integrator and Extended Kalman Filter.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.navigation.ekf import ExtendedKalmanFilter
from rocketpy.gnc.navigation.imu_integration import IMUIntegrator
from rocketpy.gnc.navigation.quaternion import Quaternion
from rocketpy.gnc.navigation.sensor_packet import SensorPacket


class SensorPipeline:
    """
    High-level navigation processing pipeline.

    The pipeline coordinates IMU propagation and EKF measurement
    updates to estimate the rocket navigation state.
    """

    def __init__(
        self,
        initial_position: NDArray[np.float64]
        | list[float]
        | tuple[float, ...]
        | None = None,
        initial_velocity: NDArray[np.float64]
        | list[float]
        | tuple[float, ...]
        | None = None,
        initial_attitude: Quaternion | None = None,
        gravity: NDArray[np.float64]
        | list[float]
        | tuple[float, ...]
        | None = None,
    ) -> None:
        """Initialize the navigation pipeline."""

        if initial_position is None:
            initial_position = np.zeros(3, dtype=np.float64)

        if initial_velocity is None:
            initial_velocity = np.zeros(3, dtype=np.float64)

        if initial_attitude is None:
            initial_attitude = Quaternion.identity()

        if gravity is None:
            gravity = np.array(
                [0.0, 0.0, -9.81],
                dtype=np.float64,
            )

        self._integrator = IMUIntegrator(
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            initial_attitude=initial_attitude,
            gravity=gravity,
        )

        self._ekf = ExtendedKalmanFilter()

        self._last_timestamp: float | None = None

    def update(
        self,
        packet: SensorPacket,
    ) -> NDArray[np.float64]:
        """
        Process a sensor packet and update the navigation state.

        Parameters
        ----------
        packet : SensorPacket
            Current sensor measurements.

        Returns
        -------
        numpy.ndarray
            Updated EKF state vector.
        """

        if not isinstance(packet, SensorPacket):
            raise TypeError(
                "packet must be a SensorPacket instance."
            )

        if self._last_timestamp is None:
            dt = 0.01
        else:
            dt = packet.timestamp - self._last_timestamp

        if dt <= 0.0:
            raise ValueError(
                "Sensor packet timestamps must be strictly increasing."
            )

        self._last_timestamp = packet.timestamp

        self._integrator.update(
            gyro=packet.gyroscope,
            accel=packet.accelerometer,
            dt=dt,
        )

        self._ekf.predict(
            gyro=packet.gyroscope,
            accel=packet.accelerometer,
            dt=dt,
        )

        self._ekf.update_gnss(
            packet.gps_position,
        )

        self._ekf.update_barometer(
            packet.barometric_altitude,
        )

        return self._ekf.get_state()

    def get_state(self) -> NDArray[np.float64]:
        """
        Return the current estimated navigation state.
        """
        return self._ekf.get_state()

    def reset(self) -> None:
        """
        Reset the navigation pipeline.
        """
        self._integrator.reset()
        self._ekf.reset()
        self._last_timestamp = None

__all__ = [
    "SensorPipeline",
]