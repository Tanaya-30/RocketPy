"""Coordinate frame transformation utilities for RocketGNC.

This module provides numerically stable coordinate frame transformations for
navigation, IMU integration, sensor fusion, and extended Kalman filtering.
The implementation uses the Hamilton quaternion convention, active rotations,
and a right-handed coordinate system.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rocketpy.gnc.navigation.quaternion import Quaternion

EPS = 1e-12
ORTHOGONALITY_TOL = 1e-10
WGS84_A = np.float64(6378137.0)
WGS84_B = np.float64(6356752.314245179)
WGS84_E2 = np.float64(1.0 - (WGS84_B / WGS84_A) ** 2)


def _validate_quaternion(quaternion: Any) -> Quaternion:
    """
    Validate and return a quaternion instance.

    The validated quaternion object is returned
    unchanged for subsequent frame transformations.

    Parameters
    ----------
    quaternion : Any
        Object to validate as a quaternion.

    Returns
    -------
    Quaternion
        The validated quaternion instance.

    Raises
    ------
    TypeError
        If the provided object is not a Quaternion.
    """
    if not isinstance(quaternion, Quaternion):
        raise TypeError("Expected a Quaternion instance.")
    return quaternion


def _validate_geodetic_coordinates(
    latitude: float | np.floating[Any],
    longitude: float | np.floating[Any],
) -> tuple[np.float64, np.float64]:
    """
    Validate geodetic coordinates.

    Returns the validated latitude and longitude
    converted to float64 values.
    """
    lat = np.float64(latitude)
    lon = np.float64(longitude)
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise ValueError("Latitude and longitude must be finite values.")
    if lat < -90.0 or lat > 90.0:
        raise ValueError("Latitude must be within [-90, 90] degrees.")
    if lon < -180.0 or lon > 180.0:
        raise ValueError("Longitude must be within [-180, 180] degrees.")
    return lat, lon


def validate_vector(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
) -> NDArray[np.float64]:
    """
    Validate a three-dimensional vector.

    Returns the validated vector converted to a
    float64 NumPy array.

        Parameters
        ----------
        vector : array-like of shape (3,)
            A Cartesian vector in any supported coordinate frame.

        Returns
        -------
        numpy.ndarray
            A float64 array of shape (3,).

        Raises
        ------
        ValueError
            If the vector does not contain exactly three finite values.
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != 3:
        raise ValueError("Vector must contain exactly three components.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Vector must contain only finite values.")
    return np.asarray(values, dtype=np.float64)


def validate_rotation_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate a rotation matrix.

    Parameters
    ----------
    matrix : array-like of shape (3, 3)
        A candidate rotation matrix.

    Returns
    -------
    numpy.ndarray
        A validated float64 rotation matrix.

    Raises
    ------
    ValueError
        If the input is not a 3x3 matrix or is not orthogonal within tolerance.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rotation matrix must contain only finite values.")

    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(values.T @ values, identity, atol=ORTHOGONALITY_TOL, rtol=0.0):
        raise ValueError("Rotation matrix is not orthogonal within tolerance.")
    determinant = np.linalg.det(values)
    if not np.isclose(determinant, 1.0, atol=ORTHOGONALITY_TOL, rtol=0.0):
        raise ValueError("Rotation matrix determinant must be close to 1.")
    return values.astype(np.float64, copy=False)


def enu_to_ned(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
) -> NDArray[np.float64]:
    """
    Convert a vector from the ENU frame to the NED frame.

    The returned vector is computed by swapping the
    East/North components and inverting the Up axis.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the East-North-Up frame.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the North-East-Down frame.
    """
    values = validate_vector(vector)
    return np.array([values[1], values[0], -values[2]], dtype=np.float64)


def ned_to_enu(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
) -> NDArray[np.float64]:
    """Convert a vector from the NED frame to the ENU frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the North-East-Down frame.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the East-North-Up frame.
    """
    values = validate_vector(vector)
    return np.array([values[1], values[0], -values[2]], dtype=np.float64)


def ned_to_ecef(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    latitude: float | np.floating[Any],
    longitude: float | np.floating[Any],
) -> NDArray[np.float64]:
    """Convert a vector from the NED frame to the ECEF frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the North-East-Down frame.
    latitude : float
        Geodetic latitude in degrees.
    longitude : float
        Geodetic longitude in degrees.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the ECEF frame.
    """
    values = validate_vector(vector)
    lat, lon = _validate_geodetic_coordinates(latitude, longitude)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    x = (
        -sin_lat * cos_lon * values[0]
        - sin_lon * values[1]
        - cos_lat * cos_lon * values[2]
    )
    y = (
        -sin_lat * sin_lon * values[0]
        + cos_lon * values[1]
        - cos_lat * sin_lon * values[2]
    )
    z = cos_lat * values[0] - sin_lat * values[2]
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_ned(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    latitude: float | np.floating[Any],
    longitude: float | np.floating[Any],
) -> NDArray[np.float64]:
    """Convert a vector from the ECEF frame to the NED frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the ECEF frame.
    latitude : float
        Geodetic latitude in degrees.
    longitude : float
        Geodetic longitude in degrees.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the NED frame.
    """
    values = validate_vector(vector)
    lat, lon = _validate_geodetic_coordinates(latitude, longitude)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    north = (
        -sin_lat * cos_lon * values[0]
        - sin_lat * sin_lon * values[1]
        + cos_lat * values[2]
    )
    east = -sin_lon * values[0] + cos_lon * values[1]
    down = (
        -cos_lat * cos_lon * values[0]
        - cos_lat * sin_lon * values[1]
        - sin_lat * values[2]
    )
    return np.array([north, east, down], dtype=np.float64)


def body_to_ned(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    quaternion: Quaternion,
) -> NDArray[np.float64]:
    """Rotate a vector from the body frame to the NED frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the body frame.
    quaternion : Quaternion
        Quaternion representing the active rotation from body to NED.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the NED frame.
    """
    values = validate_vector(vector)
    rotation = _validate_quaternion(quaternion)
    return np.asarray(rotation.rotate_vector(values), dtype=np.float64)


def ned_to_body(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    quaternion: Quaternion,
) -> NDArray[np.float64]:
    """Rotate a vector from the NED frame to the body frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the NED frame.
    quaternion : Quaternion
        Quaternion representing the active rotation from body to NED.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the body frame.
    """
    values = validate_vector(vector)
    rotation = _validate_quaternion(quaternion)
    return np.asarray(rotation.inverse().rotate_vector(values), dtype=np.float64)


def body_to_ecef(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    quaternion: Quaternion,
    latitude: float | np.floating[Any],
    longitude: float | np.floating[Any],
) -> NDArray[np.float64]:
    """Rotate a vector from the body frame to the ECEF frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the body frame.
    quaternion : Quaternion
        Quaternion representing the active rotation from body to NED.
    latitude : float, optional
        Geodetic latitude in degrees used for the NED-to-ECEF conversion.
    longitude : float, optional
        Geodetic longitude in degrees used for the NED-to-ECEF conversion.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the ECEF frame.
    """
    values = validate_vector(vector)
    rotation = _validate_quaternion(quaternion)
    ned_vector = rotation.rotate_vector(values)
    return ned_to_ecef(ned_vector, latitude, longitude)


def ecef_to_body(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    quaternion: Quaternion,
    latitude: float | np.floating[Any],
    longitude: float | np.floating[Any],
) -> NDArray[np.float64]:
    """Rotate a vector from the ECEF frame to the body frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the ECEF frame.
    quaternion : Quaternion
        Quaternion representing the active rotation from body to NED.
    latitude : float, optional
        Geodetic latitude in degrees used for the ECEF-to-NED conversion.
    longitude : float, optional
        Geodetic longitude in degrees used for the ECEF-to-NED conversion.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the body frame.
    """
    values = validate_vector(vector)
    rotation = _validate_quaternion(quaternion)
    ned_vector = ecef_to_ned(values, latitude, longitude)
    return np.asarray(rotation.inverse().rotate_vector(ned_vector), dtype=np.float64)


def sensor_to_body(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    sensor_rotation: Quaternion,
) -> NDArray[np.float64]:
    """Rotate a vector from the sensor frame to the body frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the sensor frame.
    sensor_rotation : Quaternion
        Quaternion representing the active rotation from sensor to body.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the body frame.
    """
    values = validate_vector(vector)
    rotation = _validate_quaternion(sensor_rotation)
    return np.asarray(rotation.rotate_vector(values), dtype=np.float64)


def body_to_sensor(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    sensor_rotation: Quaternion,
) -> NDArray[np.float64]:
    """Rotate a vector from the body frame to the sensor frame.

    Parameters
    ----------
    vector : array-like of shape (3,)
        Vector expressed in the body frame.
    sensor_rotation : Quaternion
        Quaternion representing the active rotation from sensor to body.

    Returns
    -------
    numpy.ndarray
        Vector expressed in the sensor frame.
    """
    values = validate_vector(vector)
    rotation = _validate_quaternion(sensor_rotation)
    return np.asarray(rotation.inverse().rotate_vector(values), dtype=np.float64)


def rotation_matrix_from_quaternion(quaternion: Quaternion) -> NDArray[np.float64]:
    """Convert a quaternion to a rotation matrix.

    Parameters
    ----------
    quaternion : Quaternion
        Quaternion following the Hamilton convention.

    Returns
    -------
    numpy.ndarray
        A float64 rotation matrix of shape (3, 3).
    """
    rotation = _validate_quaternion(quaternion)
    return np.asarray(rotation.to_rotation_matrix(), dtype=np.float64)


__all__ = [
    "Quaternion",
    "EPS",
    "ORTHOGONALITY_TOL",
    "WGS84_A",
    "WGS84_B",
    "WGS84_E2",
    "validate_vector",
    "validate_rotation_matrix",
    "enu_to_ned",
    "ned_to_enu",
    "ned_to_ecef",
    "ecef_to_ned",
    "body_to_ned",
    "ned_to_body",
    "body_to_ecef",
    "ecef_to_body",
    "sensor_to_body",
    "body_to_sensor",
    "rotation_matrix_from_quaternion",
]
