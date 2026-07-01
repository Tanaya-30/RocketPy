"""Validation utilities for RocketGNC navigation.

This module provides reusable validation functions for navigation vectors,
matrices, scalars, covariance matrices, and other common navigation inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_vector(
    vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    *,
    size: int = 3,
    name: str = "vector",
) -> NDArray[np.float64]:
    """Validate a finite vector.

    Parameters
    ----------
    vector : array-like
        Input vector.
    size : int, default=3
        Expected number of components.
    name : str, default="vector"
        Name used in exception messages.

    Returns
    -------
    numpy.ndarray
        Validated float64 vector.

    Raises
    ------
    ValueError
        If the vector has the wrong size or contains non-finite values.
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)

    if values.size != size:
        raise ValueError(
            f"{name} must contain exactly {size} components."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return values.copy()


def validate_matrix(
    matrix: NDArray[np.float64] | list[list[float]],
    *,
    shape: tuple[int, int],
    name: str = "matrix",
) -> NDArray[np.float64]:
    """Validate a finite matrix.

    Parameters
    ----------
    matrix : array-like
        Input matrix.
    shape : tuple[int, int]
        Required matrix shape.
    name : str, default="matrix"
        Name used in exception messages.

    Returns
    -------
    numpy.ndarray
        Validated float64 matrix.

    Raises
    ------
    ValueError
        If the matrix shape is incorrect or contains non-finite values.
    """
    values = np.asarray(matrix, dtype=np.float64)

    if values.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return values.copy()

def validate_quaternion(
    quaternion: Any,
    *,
    name: str = "quaternion",
) -> Any:
    """Validate a Quaternion instance.

    Parameters
    ----------
    quaternion : Any
        Quaternion object.
    name : str, default="quaternion"
        Name used in exception messages.

    Returns
    -------
    Quaternion
        Validated quaternion.

    Raises
    ------
    TypeError
        If the object is not a Quaternion instance.
    ValueError
        If the quaternion is not normalized.
    """
    from rocketpy.gnc.navigation.quaternion import Quaternion

    if not isinstance(quaternion, Quaternion):
        raise TypeError(
            f"{name} must be a Quaternion instance."
        )

    if not quaternion.is_normalized():
        raise ValueError(
            f"{name} must be normalized."
        )

    return quaternion


def validate_dt(
    dt: float | np.floating[Any],
) -> np.float64:
    """Validate an integration time step.

    Parameters
    ----------
    dt : float
        Time step in seconds.

    Returns
    -------
    numpy.float64
        Validated time step.

    Raises
    ------
    ValueError
        If the value is non-finite or not positive.
    """
    value = np.float64(dt)

    if not np.isfinite(value):
        raise ValueError(
            "dt must be finite."
        )

    if value <= 0.0:
        raise ValueError(
            "dt must be greater than zero."
        )

    return value


def validate_timestamp(
    timestamp: float | np.floating[Any],
) -> np.float64:
    """Validate a timestamp.

    Parameters
    ----------
    timestamp : float
        Timestamp in seconds.

    Returns
    -------
    numpy.float64
        Validated timestamp.

    Raises
    ------
    ValueError
        If the timestamp is non-finite or negative.
    """
    value = np.float64(timestamp)

    if not np.isfinite(value):
        raise ValueError(
            "timestamp must be finite."
        )

    if value < 0.0:
        raise ValueError(
            "timestamp must be non-negative."
        )

    return value


def validate_covariance(
    covariance: NDArray[np.float64] | list[list[float]],
    *,
    size: int,
    name: str = "covariance",
) -> NDArray[np.float64]:
    """Validate a covariance matrix.

    Parameters
    ----------
    covariance : array-like
        Covariance matrix.
    size : int
        Expected matrix dimension.
    name : str, default="covariance"
        Name used in exception messages.

    Returns
    -------
    numpy.ndarray
        Validated covariance matrix.

    Raises
    ------
    ValueError
        If the covariance matrix is invalid.
    """
    values = validate_matrix(
        covariance,
        shape=(size, size),
        name=name,
    )

    if not np.allclose(values, values.T, atol=1e-12):
        raise ValueError(
            f"{name} must be symmetric."
        )

    if np.any(np.diag(values) < 0.0):
        raise ValueError(
            f"{name} diagonal entries must be non-negative."
        )

    return values.copy()

def validate_position(
    position: NDArray[np.float64] | list[float] | tuple[float, ...],
    ) -> NDArray[np.float64]:
        """Validate a navigation-frame position vector.

        Parameters
        ----------
        position : array-like of shape (3,)
            Position vector.

        Returns
        -------
        numpy.ndarray
            Validated position vector.
        """
        return validate_vector(
            position,
            size=3,
            name="position",
        )


def validate_velocity(
    velocity: NDArray[np.float64] | list[float] | tuple[float, ...],
) -> NDArray[np.float64]:
    """Validate a navigation-frame velocity vector.

    Parameters
    ----------
    velocity : array-like of shape (3,)
        Velocity vector.

    Returns
    -------
    numpy.ndarray
        Validated velocity vector.
    """
    return validate_vector(
        velocity,
        size=3,
        name="velocity",
    )


def validate_acceleration(
    acceleration: NDArray[np.float64] | list[float] | tuple[float, ...],
) -> NDArray[np.float64]:
    """Validate an acceleration vector.

    Parameters
    ----------
    acceleration : array-like of shape (3,)
        Linear acceleration vector.

    Returns
    -------
    numpy.ndarray
        Validated acceleration vector.
    """
    return validate_vector(
        acceleration,
        size=3,
        name="acceleration",
    )


def validate_angular_velocity(
    angular_velocity: NDArray[np.float64]
    | list[float]
    | tuple[float, ...],
) -> NDArray[np.float64]:
    """Validate an angular velocity vector.

    Parameters
    ----------
    angular_velocity : array-like of shape (3,)
        Angular velocity in rad/s.

    Returns
    -------
    numpy.ndarray
        Validated angular velocity vector.
    """
    return validate_vector(
        angular_velocity,
        size=3,
        name="angular_velocity",
    )

def validate_latitude(
    latitude: float | np.floating[Any],
) -> np.float64:
    """Validate a geodetic latitude.

    Parameters
    ----------
    latitude : float
        Latitude in degrees.

    Returns
    -------
    numpy.float64
        Validated latitude.

    Raises
    ------
    ValueError
        If latitude is non-finite or outside [-90, 90].
    """
    value = np.float64(latitude)

    if not np.isfinite(value):
        raise ValueError("latitude must be finite.")

    if not (-90.0 <= value <= 90.0):
        raise ValueError(
            "latitude must be between -90 and 90 degrees."
        )

    return value


def validate_longitude(
    longitude: float | np.floating[Any],
) -> np.float64:
    """Validate a geodetic longitude.

    Parameters
    ----------
    longitude : float
        Longitude in degrees.

    Returns
    -------
    numpy.float64
        Validated longitude.

    Raises
    ------
    ValueError
        If longitude is non-finite or outside [-180, 180].
    """
    value = np.float64(longitude)

    if not np.isfinite(value):
        raise ValueError("longitude must be finite.")

    if not (-180.0 <= value <= 180.0):
        raise ValueError(
            "longitude must be between -180 and 180 degrees."
        )

    return value


def validate_altitude(
    altitude: float | np.floating[Any],
) -> np.float64:
    """Validate an altitude measurement.

    Parameters
    ----------
    altitude : float
        Altitude in metres.

    Returns
    -------
    numpy.float64
        Validated altitude.

    Raises
    ------
    ValueError
        If altitude is non-finite.
    """
    value = np.float64(altitude)

    if not np.isfinite(value):
        raise ValueError("altitude must be finite.")

    return value


def validate_pressure(
    pressure: float | np.floating[Any],
) -> np.float64:
    """Validate an atmospheric pressure measurement.

    Parameters
    ----------
    pressure : float
        Pressure in pascals.

    Returns
    -------
    numpy.float64
        Validated pressure.

    Raises
    ------
    ValueError
        If pressure is non-finite or not positive.
    """
    value = np.float64(pressure)

    if not np.isfinite(value):
        raise ValueError("pressure must be finite.")

    if value <= 0.0:
        raise ValueError(
            "pressure must be greater than zero."
        )

    return value


__all__ = [
    "validate_vector",
    "validate_matrix",
    "validate_quaternion",
    "validate_dt",
    "validate_timestamp",
    "validate_covariance",
    "validate_position",
    "validate_velocity",
    "validate_acceleration",
    "validate_angular_velocity",
    "validate_latitude",
    "validate_longitude",
    "validate_altitude",
    "validate_pressure",
]