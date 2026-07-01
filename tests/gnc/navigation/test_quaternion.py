import numpy as np
import pytest

from rocketpy.gnc.navigation.quaternion import Quaternion


def test_identity_and_properties() -> None:
    q = Quaternion.identity()

    assert q.w == 1.0
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0
    assert q.as_numpy().dtype == np.float64


def test_euler_round_trip() -> None:
    roll, pitch, yaw = np.deg2rad([10.0, 20.0, 30.0])
    q = Quaternion.from_euler(roll, pitch, yaw)

    recovered_roll, recovered_pitch, recovered_yaw = q.to_euler()
    assert np.allclose(
        [recovered_roll, recovered_pitch, recovered_yaw],
        [roll, pitch, yaw],
        atol=1e-10,
    )


def test_rotate_vector_about_z_axis() -> None:
    q = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float64), np.pi / 2.0)
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    rotated = q.rotate_vector(vector)
    assert np.allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_integrate_preserves_unit_norm() -> None:
    q = Quaternion.identity()
    q.integrate(np.array([0.0, 0.0, 0.1], dtype=np.float64), 0.1)

    assert np.isclose(q.norm(), 1.0, atol=1e-12)


def test_normalize_and_is_normalized() -> None:
    q = Quaternion(2.0, 0.0, 0.0, 0.0)
    q.normalize()

    assert np.isclose(q.norm(), 1.0, atol=1e-12)
    assert q.is_normalized()


def test_normalized_returns_unit_copy() -> None:
    q = Quaternion(2.0, 0.0, 0.0, 0.0)
    normalized = q.normalized()

    assert normalized is not q
    assert np.isclose(normalized.norm(), 1.0, atol=1e-12)


def test_conjugate_and_inverse() -> None:
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    conjugate = q.conjugate()
    inverse = q.inverse()

    assert np.allclose(conjugate.as_numpy(), np.array([1.0, -2.0, -3.0, -4.0]))
    assert np.allclose((q * inverse).as_numpy(), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-12)


def test_quaternion_multiplication_and_scalar_ops() -> None:
    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(5.0, 6.0, 7.0, 8.0)

    product = q1 * q2
    assert np.allclose(
        product.as_numpy(),
        np.array([ -60.0, 12.0, 30.0, 24.0 ], dtype=np.float64),
    )

    scaled = q1 * 2.0
    assert np.allclose(scaled.as_numpy(), np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64))

    left_scaled = 2.0 * q1
    assert np.allclose(left_scaled.as_numpy(), np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64))

    divided = q1 / 2.0
    assert np.allclose(divided.as_numpy(), np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64))


def test_dot_product() -> None:
    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(5.0, 6.0, 7.0, 8.0)

    assert np.isclose(q1.dot(q2), 70.0, atol=1e-12)


def test_from_numpy_and_from_rotation_matrix_round_trip() -> None:
    values = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    q = Quaternion.from_numpy(values)
    assert np.allclose(q.as_numpy(), values)

    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    q_from_rotation = Quaternion.from_rotation_matrix(rotation)
    recovered = q_from_rotation.to_rotation_matrix()
    assert np.allclose(recovered, rotation, atol=1e-12)


def test_from_axis_angle_and_to_axis_angle() -> None:
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    q = Quaternion.from_axis_angle(axis, np.pi / 2.0)
    recovered_axis, recovered_angle = q.to_axis_angle()

    assert np.allclose(recovered_axis, axis, atol=1e-12)
    assert np.isclose(recovered_angle, np.pi / 2.0, atol=1e-12)


def test_rotate_vector_and_integration() -> None:
    q = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float64), np.pi / 2.0)
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rotated = q.rotate_vector(vector)
    assert np.allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-12)

    q_integrated = Quaternion.identity()
    q_integrated.integrate(np.array([0.0, 0.0, 0.1], dtype=np.float64), 0.1)
    assert np.isclose(q_integrated.norm(), 1.0, atol=1e-12)


def test_invalid_axis_angle() -> None:
    with pytest.raises(ValueError):
        Quaternion.from_axis_angle(np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0)


def test_invalid_rotation_matrix() -> None:
    with pytest.raises(ValueError):
        Quaternion.from_rotation_matrix(np.eye(3, dtype=np.float64) * 2.0)


def test_invalid_vector_size() -> None:
    with pytest.raises(ValueError):
        Quaternion.identity().rotate_vector(np.array([1.0, 0.0], dtype=np.float64))


def test_invalid_angular_velocity() -> None:
    with pytest.raises(ValueError):
        Quaternion.identity().integrate(np.array([0.0, 0.0], dtype=np.float64), 0.1)


def test_zero_quaternion_normalization() -> None:
    q = Quaternion(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        q.normalize()


def test_zero_quaternion_inversion() -> None:
    q = Quaternion(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        q.inverse()


def test_zero_scalar_division() -> None:
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ZeroDivisionError):
        q / 0.0
