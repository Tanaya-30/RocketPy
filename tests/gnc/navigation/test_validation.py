import numpy as np
import pytest

from rocketpy.gnc.navigation.quaternion import Quaternion
from rocketpy.gnc.navigation.validation import (
    validate_acceleration,
    validate_altitude,
    validate_angular_velocity,
    validate_covariance,
    validate_dt,
    validate_latitude,
    validate_longitude,
    validate_matrix,
    validate_position,
    validate_pressure,
    validate_quaternion,
    validate_timestamp,
    validate_vector,
    validate_velocity,
)

def test_validate_vector_accepts_numpy_array() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = validate_vector(vector)

    assert np.allclose(result, vector)
    assert result.dtype == np.float64


def test_validate_vector_accepts_list() -> None:
    result = validate_vector([1.0, 2.0, 3.0])

    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_validate_vector_accepts_tuple() -> None:
    result = validate_vector((1.0, 2.0, 3.0))

    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_validate_vector_returns_copy() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = validate_vector(vector)

    result[0] = 99.0

    assert vector[0] == 1.0


@pytest.mark.parametrize(
    "vector",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        np.zeros(2),
        np.zeros(4),
    ],
)
def test_validate_vector_rejects_wrong_size(vector) -> None:
    with pytest.raises(ValueError):
        validate_vector(vector)


@pytest.mark.parametrize(
    "vector",
    [
        [1.0, np.nan, 3.0],
        [1.0, np.inf, 3.0],
        [1.0, -np.inf, 3.0],
    ],
)
def test_validate_vector_rejects_non_finite_values(vector) -> None:
    with pytest.raises(ValueError):
        validate_vector(vector)


def test_validate_matrix_accepts_valid_matrix() -> None:
    matrix = np.eye(3, dtype=np.float64)

    result = validate_matrix(matrix, shape=(3, 3))

    assert np.allclose(result, matrix)
    assert result.dtype == np.float64


def test_validate_matrix_returns_copy() -> None:
    matrix = np.eye(3, dtype=np.float64)

    result = validate_matrix(matrix, shape=(3, 3))

    result[0, 0] = 99.0

    assert matrix[0, 0] == 1.0


def test_validate_matrix_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        validate_matrix(np.eye(2), shape=(3, 3))


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, np.nan], [0.0, 1.0]],
        [[1.0, np.inf], [0.0, 1.0]],
    ],
)
def test_validate_matrix_rejects_non_finite_values(matrix) -> None:
    with pytest.raises(ValueError):
        validate_matrix(matrix, shape=(2, 2))

def test_validate_quaternion_accepts_normalized_quaternion() -> None:
    quaternion = Quaternion.identity()

    result = validate_quaternion(quaternion)

    assert result is quaternion


def test_validate_quaternion_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        validate_quaternion([1.0, 0.0, 0.0, 0.0])


def test_validate_quaternion_rejects_non_normalized_quaternion() -> None:
    quaternion = Quaternion(2.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        validate_quaternion(quaternion)


def test_validate_dt_accepts_positive_value() -> None:
    result = validate_dt(0.1)

    assert result == pytest.approx(0.1)
    assert isinstance(result, np.float64)

@pytest.mark.parametrize("dt", [0.0, -0.1, -1.0])
def test_validate_dt_rejects_non_positive_values(dt: float) -> None:
    with pytest.raises(ValueError):
        validate_dt(dt)


@pytest.mark.parametrize("dt", [np.nan, np.inf, -np.inf])
def test_validate_dt_rejects_non_finite_values(dt: float) -> None:
    with pytest.raises(ValueError):
        validate_dt(dt)


def test_validate_timestamp_accepts_zero() -> None:
    result = validate_timestamp(0.0)

    assert result == pytest.approx(0.0)
    assert isinstance(result, np.float64)


def test_validate_timestamp_accepts_positive_value() -> None:
    result = validate_timestamp(12.5)

    assert result == pytest.approx(12.5)


def test_validate_timestamp_rejects_negative_value() -> None:
    with pytest.raises(ValueError):
        validate_timestamp(-0.001)


@pytest.mark.parametrize("timestamp", [np.nan, np.inf, -np.inf])
def test_validate_timestamp_rejects_non_finite_values(
    timestamp: float,
) -> None:
    with pytest.raises(ValueError):
        validate_timestamp(timestamp)

def test_validate_covariance_accepts_valid_matrix() -> None:
    covariance = np.eye(6, dtype=np.float64)

    result = validate_covariance(covariance, size=6)

    assert np.allclose(result, covariance)


def test_validate_covariance_returns_copy() -> None:
    covariance = np.eye(4, dtype=np.float64)

    result = validate_covariance(covariance, size=4)

    result[0, 0] = 10.0

    assert covariance[0, 0] == 1.0


def test_validate_covariance_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        validate_covariance(np.eye(3), size=4)


def test_validate_covariance_rejects_non_symmetric_matrix() -> None:
    covariance = np.eye(3, dtype=np.float64)
    covariance[0, 1] = 1.0

    with pytest.raises(ValueError):
        validate_covariance(covariance, size=3)


def test_validate_covariance_rejects_negative_diagonal() -> None:
    covariance = np.eye(3, dtype=np.float64)
    covariance[1, 1] = -1.0

    with pytest.raises(ValueError):
        validate_covariance(covariance, size=3)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([1.0, 2.0, 3.0]),
        [1.0, 2.0, 3.0],
        (1.0, 2.0, 3.0),
    ],
)
def test_validate_position_accepts_valid_vectors(vector) -> None:
    result = validate_position(vector)

    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_validate_position_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        validate_position([1.0, 2.0])


def test_validate_velocity_accepts_valid_vector() -> None:
    result = validate_velocity([4.0, 5.0, 6.0])

    assert np.allclose(result, [4.0, 5.0, 6.0])


def test_validate_velocity_rejects_nan() -> None:
    with pytest.raises(ValueError):
        validate_velocity([1.0, np.nan, 3.0])


def test_validate_acceleration_accepts_valid_vector() -> None:
    result = validate_acceleration([0.1, 0.2, 0.3])

    assert np.allclose(result, [0.1, 0.2, 0.3])


def test_validate_acceleration_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        validate_acceleration([1.0, 2.0])


def test_validate_angular_velocity_accepts_valid_vector() -> None:
    result = validate_angular_velocity([0.5, 0.6, 0.7])

    assert np.allclose(result, [0.5, 0.6, 0.7])


def test_validate_angular_velocity_rejects_inf() -> None:
    with pytest.raises(ValueError):
        validate_angular_velocity([1.0, np.inf, 3.0])


def test_validate_latitude_accepts_valid_values() -> None:
    assert validate_latitude(-90.0) == pytest.approx(-90.0)
    assert validate_latitude(0.0) == pytest.approx(0.0)
    assert validate_latitude(90.0) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "latitude",
    [-90.1, 90.1, np.nan, np.inf, -np.inf],
)
def test_validate_latitude_rejects_invalid_values(
    latitude: float,
) -> None:
    with pytest.raises(ValueError):
        validate_latitude(latitude)


def test_validate_longitude_accepts_valid_values() -> None:
    assert validate_longitude(-180.0) == pytest.approx(-180.0)
    assert validate_longitude(0.0) == pytest.approx(0.0)
    assert validate_longitude(180.0) == pytest.approx(180.0)


@pytest.mark.parametrize(
    "longitude",
    [-180.1, 180.1, np.nan, np.inf, -np.inf],
)
def test_validate_longitude_rejects_invalid_values(
    longitude: float,
) -> None:
    with pytest.raises(ValueError):
        validate_longitude(longitude)


def test_validate_altitude_accepts_valid_values() -> None:
    assert validate_altitude(0.0) == pytest.approx(0.0)
    assert validate_altitude(1000.0) == pytest.approx(1000.0)
    assert validate_altitude(-430.0) == pytest.approx(-430.0)


@pytest.mark.parametrize(
    "altitude",
    [np.nan, np.inf, -np.inf],
)
def test_validate_altitude_rejects_non_finite_values(
    altitude: float,
) -> None:
    with pytest.raises(ValueError):
        validate_altitude(altitude)


def test_validate_pressure_accepts_positive_value() -> None:
    result = validate_pressure(101325.0)

    assert result == pytest.approx(101325.0)
    assert isinstance(result, np.float64)


@pytest.mark.parametrize(
    "pressure",
    [0.0, -1.0, -100.0],
)
def test_validate_pressure_rejects_non_positive_values(
    pressure: float,
) -> None:
    with pytest.raises(ValueError):
        validate_pressure(pressure)


@pytest.mark.parametrize(
    "pressure",
    [np.nan, np.inf, -np.inf],
)
def test_validate_pressure_rejects_non_finite_values(
    pressure: float,
) -> None:
    with pytest.raises(ValueError):
        validate_pressure(pressure)


def test_validation_helpers_return_float64_scalars() -> None:
    assert isinstance(validate_dt(0.1), np.float64)
    assert isinstance(validate_timestamp(1.0), np.float64)
    assert isinstance(validate_latitude(10.0), np.float64)
    assert isinstance(validate_longitude(20.0), np.float64)
    assert isinstance(validate_altitude(100.0), np.float64)
    assert isinstance(validate_pressure(101325.0), np.float64)