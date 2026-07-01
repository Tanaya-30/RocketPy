
import numpy as np
import pytest

from rocketpy.gnc.navigation.ekf import ExtendedKalmanFilter
from rocketpy.gnc.navigation.quaternion import Quaternion


def test_constructor_creates_valid_default_state() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.get_state()

    assert state.shape == (13,)
    assert state.dtype == np.float64
    assert np.all(np.isfinite(state))
    assert np.isclose(np.linalg.norm(state[6:10]), 1.0, atol=1e-12)


def test_constructor_creates_valid_default_covariance() -> None:
    ekf = ExtendedKalmanFilter()

    covariance = ekf.get_covariance()

    assert covariance.shape == (13, 13)
    assert covariance.dtype == np.float64
    assert np.allclose(covariance, covariance.T, atol=1e-12)


def test_get_state_returns_copy() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.get_state()
    state[0] = 999.0

    assert ekf.get_state()[0] != 999.0


def test_get_covariance_returns_copy() -> None:
    ekf = ExtendedKalmanFilter()

    covariance = ekf.get_covariance()
    covariance[0, 0] = 999.0

    assert ekf.get_covariance()[0, 0] != 999.0


def test_set_state_accepts_valid_state() -> None:
    ekf = ExtendedKalmanFilter()

    state = np.zeros(13, dtype=np.float64)
    state[6] = 1.0

    ekf.set_state(state)

    assert np.allclose(ekf.get_state(), state)


def test_set_state_rejects_invalid_shape() -> None:
    ekf = ExtendedKalmanFilter()

    with pytest.raises(ValueError):
        ekf.set_state(np.zeros(12, dtype=np.float64))


def test_set_state_rejects_non_finite_values() -> None:
    ekf = ExtendedKalmanFilter()

    state = np.zeros(13, dtype=np.float64)
    state[6] = 1.0
    state[0] = np.nan

    with pytest.raises(ValueError):
        ekf.set_state(state)


def test_set_covariance_accepts_valid_matrix() -> None:
    ekf = ExtendedKalmanFilter()

    covariance = np.eye(13, dtype=np.float64)

    ekf.set_covariance(covariance)

    assert np.allclose(ekf.get_covariance(), covariance)


def test_set_covariance_rejects_invalid_shape() -> None:
    ekf = ExtendedKalmanFilter()

    with pytest.raises(ValueError):
        ekf.set_covariance(np.eye(12, dtype=np.float64))


def test_set_covariance_rejects_non_symmetric_matrix() -> None:
    ekf = ExtendedKalmanFilter()

    covariance = np.eye(13, dtype=np.float64)
    covariance[0, 1] = 1.0

    with pytest.raises(ValueError):
        ekf.set_covariance(covariance)


def test_initial_quaternion_is_normalized() -> None:
    ekf = ExtendedKalmanFilter()

    quaternion = Quaternion.from_numpy(ekf.get_state()[6:10])

    assert quaternion.is_normalized()

def test_predict_with_zero_motion_keeps_state_finite() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.predict(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.1,
    )

    assert state.shape == (13,)
    assert np.all(np.isfinite(state))


def test_predict_changes_state_with_acceleration() -> None:
    ekf = ExtendedKalmanFilter()

    initial_state = ekf.get_state().copy()

    state = ekf.predict(
        np.zeros(3, dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        0.1,
    )

    assert not np.allclose(state, initial_state)


def test_predict_changes_attitude_with_angular_velocity() -> None:
    ekf = ExtendedKalmanFilter()

    initial_quaternion = ekf.get_state()[6:10].copy()

    state = ekf.predict(
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.1,
    )

    assert not np.allclose(state[6:10], initial_quaternion)


def test_predict_preserves_quaternion_normalization() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.predict(
        np.array([0.2, -0.1, 0.3], dtype=np.float64),
        np.array([1.0, 2.0, -1.0], dtype=np.float64),
        0.1,
    )

    assert np.isclose(np.linalg.norm(state[6:10]), 1.0, atol=1e-12)


def test_predict_keeps_covariance_symmetric() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.predict(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.1,
    )

    covariance = ekf.get_covariance()

    assert np.allclose(covariance, covariance.T, atol=1e-12)


def test_multiple_predictions_keep_quaternion_normalized() -> None:
    ekf = ExtendedKalmanFilter()

    for _ in range(20):
        state = ekf.predict(
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            np.array([0.5, -0.2, 1.0], dtype=np.float64),
            0.05,
        )

    assert np.isclose(np.linalg.norm(state[6:10]), 1.0, atol=1e-12)


@pytest.mark.parametrize("bad_dt", [0.0, -1.0])
def test_predict_rejects_non_positive_dt(bad_dt: float) -> None:
    ekf = ExtendedKalmanFilter()

    with pytest.raises(ValueError):
        ekf.predict(
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            bad_dt,
        )


@pytest.mark.parametrize("bad_dt", [np.nan, np.inf])
def test_predict_rejects_non_finite_dt(bad_dt: float) -> None:
    ekf = ExtendedKalmanFilter()

    with pytest.raises(ValueError):
        ekf.predict(
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            bad_dt,
        )


def test_predict_rejects_invalid_gyro_vector() -> None:
    ekf = ExtendedKalmanFilter()

    with pytest.raises(ValueError):
        ekf.predict(
            np.array([1.0, 2.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            0.1,
        )


def test_predict_rejects_invalid_acceleration_vector() -> None:
    ekf = ExtendedKalmanFilter()

    with pytest.raises(ValueError):
        ekf.predict(
            np.zeros(3, dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            0.1,
        )

def test_update_gnss_corrects_position_state() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.predict(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.1,
    )

    state_before = ekf.get_state().copy()

    state_after = ekf.update_gnss(
        np.array([10.0, 20.0, 30.0], dtype=np.float64)
    )

    assert state_after.shape == (13,)
    assert np.all(np.isfinite(state_after))
    assert not np.allclose(state_before[:3], state_after[:3])


def test_update_gnss_keeps_quaternion_normalized() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.update_gnss(
        np.array([1.0, 2.0, 3.0], dtype=np.float64)
    )

    assert np.isclose(np.linalg.norm(state[6:10]), 1.0, atol=1e-12)


def test_update_gnss_keeps_covariance_symmetric() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.update_gnss(
        np.array([1.0, 2.0, 3.0], dtype=np.float64)
    )

    covariance = ekf.get_covariance()

    assert np.allclose(covariance, covariance.T, atol=1e-12)


def test_multiple_gnss_updates_produce_finite_state() -> None:
    ekf = ExtendedKalmanFilter()

    for i in range(5):
        state = ekf.update_gnss(
            np.array(
                [i + 1.0, i + 2.0, i + 3.0],
                dtype=np.float64,
            )
        )

    assert np.all(np.isfinite(state))


def test_update_gnss_rejects_invalid_vector_size() -> None:
    ekf = ExtendedKalmanFilter()

    with pytest.raises(ValueError):
        ekf.update_gnss(
            np.array([1.0, 2.0], dtype=np.float64)
        )


def test_update_gnss_rejects_nan_measurement() -> None:
    ekf = ExtendedKalmanFilter()

    measurement = np.array(
        [1.0, np.nan, 3.0],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        ekf.update_gnss(measurement)


def test_update_gnss_rejects_inf_measurement() -> None:
    ekf = ExtendedKalmanFilter()

    measurement = np.array(
        [1.0, np.inf, 3.0],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        ekf.update_gnss(measurement)


def test_update_gnss_returns_copy_of_state() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.update_gnss(
        np.array([1.0, 2.0, 3.0], dtype=np.float64)
    )

    state[0] = 999.0

    assert ekf.get_state()[0] != 999.0

def test_update_barometer_corrects_altitude_state() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.predict(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.1,
    )

    state_before = ekf.get_state().copy()

    state_after = ekf.update_barometer(150.0)

    assert state_after.shape == (13,)
    assert np.all(np.isfinite(state_after))
    assert not np.isclose(state_before[2], state_after[2])


def test_update_barometer_keeps_quaternion_normalized() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.update_barometer(100.0)

    assert np.isclose(
        np.linalg.norm(state[6:10]),
        1.0,
        atol=1e-12,
    )


def test_update_barometer_keeps_covariance_symmetric() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.update_barometer(100.0)

    covariance = ekf.get_covariance()

    assert np.allclose(
        covariance,
        covariance.T,
        atol=1e-12,
    )


def test_multiple_barometer_updates_produce_finite_state() -> None:
    ekf = ExtendedKalmanFilter()

    for altitude in [100.0, 101.0, 102.0, 103.0]:
        state = ekf.update_barometer(altitude)

    assert np.all(np.isfinite(state))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_update_barometer_rejects_non_finite_measurement(
    bad_value: float,
) -> None:
    ekf = ExtendedKalmanFilter()

    with pytest.raises(ValueError):
        ekf.update_barometer(bad_value)


def test_update_barometer_returns_copy_of_state() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.update_barometer(100.0)

    state[2] = 999.0

    assert ekf.get_state()[2] != 999.0


def test_barometer_update_does_not_change_state_size() -> None:
    ekf = ExtendedKalmanFilter()

    state = ekf.update_barometer(100.0)

    assert state.shape == (13,)

def test_reset_restores_initial_state() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.predict(
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        0.1,
    )

    ekf.update_gnss(np.array([5.0, 6.0, 7.0], dtype=np.float64))
    ekf.update_barometer(100.0)

    ekf.reset()

    state = ekf.get_state()

    assert state.shape == (13,)
    assert np.allclose(state[:6], 0.0)
    assert np.allclose(state[6:10], np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(state[10:], 0.0)


def test_reset_restores_covariance() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.predict(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.1,
    )

    ekf.reset()

    covariance = ekf.get_covariance()

    assert covariance.shape == (13, 13)
    assert np.allclose(covariance, covariance.T, atol=1e-12)


def test_set_covariance_rejects_non_finite_values() -> None:
    ekf = ExtendedKalmanFilter()

    covariance = np.eye(13, dtype=np.float64)
    covariance[0, 0] = np.nan

    with pytest.raises(ValueError):
        ekf.set_covariance(covariance)


def test_set_covariance_rejects_negative_diagonal() -> None:
    ekf = ExtendedKalmanFilter()

    covariance = np.eye(13, dtype=np.float64)
    covariance[0, 0] = -1.0

    with pytest.raises(ValueError):
        ekf.set_covariance(covariance)

def test_predict_gnss_and_barometer_work_together() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.predict(
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        0.1,
    )

    ekf.update_gnss(
        np.array([10.0, 20.0, 30.0], dtype=np.float64)
    )

    state = ekf.update_barometer(35.0)

    assert state.shape == (13,)
    assert np.all(np.isfinite(state))
    assert np.isclose(np.linalg.norm(state[6:10]), 1.0, atol=1e-12)


def test_repeated_prediction_and_updates_remain_stable() -> None:
    ekf = ExtendedKalmanFilter()

    for _ in range(20):
        ekf.predict(
            np.array([0.05, 0.02, 0.01], dtype=np.float64),
            np.array([0.5, 0.0, 9.81], dtype=np.float64),
            0.05,
        )

        ekf.update_gnss(
            np.array([1.0, 2.0, 3.0], dtype=np.float64)
        )

        ekf.update_barometer(3.0)

    state = ekf.get_state()
    covariance = ekf.get_covariance()

    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(covariance))
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    assert np.isclose(np.linalg.norm(state[6:10]), 1.0, atol=1e-12)


def test_state_dimension_remains_constant() -> None:
    ekf = ExtendedKalmanFilter()

    ekf.predict(
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.1,
    )

    ekf.update_gnss(
        np.array([1.0, 2.0, 3.0], dtype=np.float64)
    )

    ekf.update_barometer(100.0)

    assert ekf.get_state().shape == (13,)