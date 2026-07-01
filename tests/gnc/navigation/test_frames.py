import numpy as np
import pytest

from rocketpy.gnc.navigation.frames import (
    EPS,
    ORTHOGONALITY_TOL,
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    body_to_ecef,
    body_to_ned,
    body_to_sensor,
    ecef_to_body,
    ecef_to_ned,
    enu_to_ned,
    ned_to_body,
    ned_to_ecef,
    ned_to_enu,
    rotation_matrix_from_quaternion,
    sensor_to_body,
    validate_rotation_matrix,
    validate_vector,
)
from rocketpy.gnc.navigation.quaternion import Quaternion


def test_validate_vector_accepts_valid_input() -> None:
    vector = validate_vector([1.0, 2.0, 3.0])

    assert vector.dtype == np.float64
    assert np.allclose(vector, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_validate_vector_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        validate_vector([1.0, 2.0])


def test_validate_vector_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        validate_vector([1.0, np.nan, 3.0])


def test_validate_rotation_matrix_accepts_valid_input() -> None:
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    assert np.allclose(validate_rotation_matrix(matrix), matrix)


def test_validate_rotation_matrix_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        validate_rotation_matrix(np.eye(2, dtype=np.float64))


def test_validate_rotation_matrix_rejects_non_orthogonal_matrix() -> None:
    with pytest.raises(ValueError):
        validate_rotation_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64))


def test_enu_to_ned_swaps_components() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert np.allclose(enu_to_ned(vector), np.array([2.0, 1.0, -3.0], dtype=np.float64))


def test_ned_to_enu_swaps_components() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert np.allclose(ned_to_enu(vector), np.array([2.0, 1.0, -3.0], dtype=np.float64))


def test_ned_to_ecef_at_equator() -> None:
    north = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    east = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    down = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    assert np.allclose(ned_to_ecef(north, 0.0, 0.0), np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert np.allclose(ned_to_ecef(east, 0.0, 0.0), np.array([0.0, 1.0, 0.0]), atol=1e-12)
    assert np.allclose(ned_to_ecef(down, 0.0, 0.0), np.array([-1.0, 0.0, 0.0]), atol=1e-12)


def test_ecef_to_ned_at_equator() -> None:
    north = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    east = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    down = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    assert np.allclose(ecef_to_ned(np.array([0.0, 0.0, 1.0]), 0.0, 0.0), north, atol=1e-12)
    assert np.allclose(ecef_to_ned(np.array([0.0, 1.0, 0.0]), 0.0, 0.0), east, atol=1e-12)
    assert np.allclose(ecef_to_ned(np.array([-1.0, 0.0, 0.0]), 0.0, 0.0), down, atol=1e-12)


def test_body_to_ned_and_ned_to_body() -> None:
    quaternion = Quaternion.identity()
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    transformed = body_to_ned(vector, quaternion)
    recovered = ned_to_body(transformed, quaternion)

    assert np.allclose(transformed, vector)
    assert np.allclose(recovered, vector)


def test_body_to_ecef_and_ecef_to_body() -> None:
    quaternion = Quaternion.identity()
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    ecef_vector = body_to_ecef(vector, quaternion, 0.0, 0.0)
    body_vector = ecef_to_body(ecef_vector, quaternion, 0.0, 0.0)

    assert np.allclose(ecef_vector, np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert np.allclose(body_vector, vector, atol=1e-12)


def test_sensor_to_body_and_body_to_sensor() -> None:
    quaternion = Quaternion.identity()
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    transformed = sensor_to_body(vector, quaternion)
    recovered = body_to_sensor(transformed, quaternion)

    assert np.allclose(transformed, vector)
    assert np.allclose(recovered, vector)


def test_rotation_matrix_from_quaternion_matches_quaternion_representation() -> None:
    quaternion = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float64), np.pi / 2.0)
    matrix = rotation_matrix_from_quaternion(quaternion)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, quaternion.to_rotation_matrix(), atol=1e-12)


def test_wgs84_constants_are_defined() -> None:
    assert WGS84_A == pytest.approx(6378137.0)
    assert WGS84_B == pytest.approx(6356752.314245179)
    assert WGS84_E2 == pytest.approx(1.0 - (WGS84_B / WGS84_A) ** 2)


def test_geodetic_validation_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ValueError):
        ned_to_ecef(np.array([1.0, 0.0, 0.0], dtype=np.float64), 91.0, 0.0)

    with pytest.raises(ValueError):
        ecef_to_ned(np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0, 181.0)


def test_round_trip_frame_transformations_preserve_values() -> None:
    quaternion = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float64), np.pi / 2.0)
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    ned_vector = body_to_ned(vector, quaternion)
    body_vector = ned_to_body(ned_vector, quaternion)
    assert np.allclose(body_vector, vector, atol=1e-12)

    ecef_vector = body_to_ecef(vector, quaternion, 0.0, 0.0)
    recovered_vector = ecef_to_body(ecef_vector, quaternion, 0.0, 0.0)
    assert np.allclose(recovered_vector, vector, atol=1e-12)


def test_frame_module_constants_are_scalar() -> None:
    assert isinstance(EPS, float)
    assert isinstance(ORTHOGONALITY_TOL, float)
