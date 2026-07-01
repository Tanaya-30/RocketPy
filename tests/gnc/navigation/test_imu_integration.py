import numpy as np
import pytest

from rocketpy.gnc.navigation.imu_integration import IMUIntegrator
from rocketpy.gnc.navigation.quaternion import Quaternion


def test_constructor_initializes_state_and_normalizes_attitude() -> None:
    initial_position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    initial_velocity = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    initial_attitude = Quaternion(2.0, 0.0, 0.0, 0.0)
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    integrator = IMUIntegrator(initial_position, initial_velocity, initial_attitude, gravity)

    assert np.allclose(integrator.position, initial_position)
    assert np.allclose(integrator.velocity, initial_velocity)
    assert np.allclose(integrator.attitude.as_numpy(), Quaternion.identity().as_numpy())
    assert integrator.attitude.is_normalized()
    assert np.allclose(integrator._gravity, gravity)


def test_update_with_zero_motion_leaves_state_unchanged() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    position, velocity, attitude = integrator.update(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.1,
    )

    assert np.allclose(position, np.zeros(3, dtype=np.float64))
    assert np.allclose(velocity, np.zeros(3, dtype=np.float64))
    assert np.allclose(attitude.as_numpy(), Quaternion.identity().as_numpy())


def test_update_with_constant_acceleration_matches_closed_form() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )
    acceleration = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    dt = 0.2

    position, velocity, _ = integrator.update(np.zeros(3, dtype=np.float64), acceleration, dt)

    expected_velocity = acceleration * dt
    expected_position = 0.5 * acceleration * dt**2
    assert np.allclose(velocity, expected_velocity)
    assert np.allclose(position, expected_position)


def test_update_with_constant_angular_velocity_changes_attitude_as_expected() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )
    gyro = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dt = 0.1

    _, _, attitude = integrator.update(gyro, np.zeros(3, dtype=np.float64), dt)

    expected_attitude = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float64), dt)
    assert np.allclose(attitude.as_numpy(), expected_attitude.as_numpy(), atol=1e-4)


def test_quaternion_remains_normalized_after_updates() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    for _ in range(5):
        integrator.update(np.array([0.0, 0.0, 1.0], dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.1)

    assert integrator.attitude.is_normalized()
    assert np.isclose(integrator.attitude.norm(), 1.0, atol=1e-12)


def test_reset_restores_initial_state() -> None:
    initial_position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    initial_velocity = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    initial_attitude = Quaternion.identity()
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    integrator = IMUIntegrator(initial_position, initial_velocity, initial_attitude, gravity)
    integrator.update(np.array([0.0, 0.0, 1.0], dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.1)
    integrator.reset()

    assert np.allclose(integrator.position, initial_position)
    assert np.allclose(integrator.velocity, initial_velocity)
    assert np.allclose(integrator.attitude.as_numpy(), initial_attitude.as_numpy())
    assert np.allclose(integrator._gravity, gravity)


def test_get_state_returns_current_state() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    position, velocity, attitude = integrator.get_state()

    assert np.allclose(position, integrator.position)
    assert np.allclose(velocity, integrator.velocity)
    assert np.allclose(attitude.as_numpy(), integrator.attitude.as_numpy())


def test_set_state_updates_navigation_state() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )
    new_position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    new_velocity = np.array([0.5, -0.5, 1.0], dtype=np.float64)
    new_attitude = Quaternion.from_euler(0.0, 0.0, np.pi / 2.0)

    integrator.set_state(new_position, new_velocity, new_attitude)

    assert np.allclose(integrator.position, new_position)
    assert np.allclose(integrator.velocity, new_velocity)
    assert np.allclose(integrator.attitude.as_numpy(), new_attitude.normalized().as_numpy())


@pytest.mark.parametrize("bad_dt", [np.nan, np.inf])
def test_update_rejects_invalid_dt(bad_dt: float) -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    with pytest.raises(ValueError):
        integrator.update(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), bad_dt)


@pytest.mark.parametrize("bad_dt", [0.0, -0.1])
def test_update_rejects_non_positive_dt(bad_dt: float) -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    with pytest.raises(ValueError):
        integrator.update(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), bad_dt)


def test_update_rejects_invalid_gyro_vector() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    with pytest.raises(ValueError):
        integrator.update(np.array([0.0, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64), 0.1)


def test_update_rejects_invalid_accel_vector() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    with pytest.raises(ValueError):
        integrator.update(np.zeros(3, dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64), 0.1)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_constructor_rejects_non_finite_values(bad_value: float) -> None:
    with pytest.raises(ValueError):
        IMUIntegrator(
            np.array([0.0, bad_value, 0.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            Quaternion.identity(),
            np.zeros(3, dtype=np.float64),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_update_rejects_non_finite_vector_components(bad_value: float) -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    with pytest.raises(ValueError):
        integrator.update(np.array([0.0, bad_value, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64), 0.1)


def test_constructor_rejects_invalid_quaternion() -> None:
    with pytest.raises(TypeError):
        IMUIntegrator(
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            object(),
            np.zeros(3, dtype=np.float64),
        )


def test_repeated_updates_accumulate_state_correctly() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    integrator.update(np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.1)
    integrator.update(np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.1)

    expected_velocity = np.array([0.2, 0.0, 0.0], dtype=np.float64)
    expected_position = np.array([0.02, 0.0, 0.0], dtype=np.float64)
    assert np.allclose(integrator.velocity, expected_velocity)
    assert np.allclose(integrator.position, expected_position)


def test_gravity_is_added_to_navigation_acceleration() -> None:
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        gravity,
    )

    position, velocity, _ = integrator.update(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.1)

    expected_velocity = gravity * 0.1
    expected_position = 0.5 * gravity * 0.1**2
    assert np.allclose(velocity, expected_velocity)
    assert np.allclose(position, expected_position)

def test_get_state_returns_copies() -> None:
    integrator = IMUIntegrator(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        Quaternion.identity(),
        np.zeros(3, dtype=np.float64),
    )

    position, velocity, attitude = integrator.get_state()

    position[0] = 100.0
    velocity[0] = 100.0

    assert integrator.position[0] == 0.0
    assert integrator.velocity[0] == 0.0

    attitude.normalize()
    assert integrator.attitude.is_normalized()
