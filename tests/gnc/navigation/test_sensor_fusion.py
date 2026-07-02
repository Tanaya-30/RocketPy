import numpy as np
import pytest

from rocketpy.gnc.navigation.sensor_fusion import SensorFusion


def test_initialization_starts_with_empty_measurements() -> None:
    fusion = SensorFusion()

    measurements = fusion.get_latest_measurements()
    assert measurements["imu"] is None
    assert measurements["gnss"] is None
    assert measurements["barometer"] is None


def test_initialization_starts_with_no_timestamps() -> None:
    fusion = SensorFusion()

    assert fusion._imu_timestamp is None
    assert fusion._gnss_timestamp is None
    assert fusion._barometer_timestamp is None


def test_update_imu_stores_valid_measurement() -> None:
    fusion = SensorFusion()
    gyro = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    accel = np.array([1.0, -1.0, 0.5], dtype=np.float64)

    fusion.update_imu(gyro, accel, 1.0)

    measurement = fusion.get_latest_measurements()["imu"]
    assert measurement is not None
    assert measurement.time == pytest.approx(1.0)
    assert measurement.gyro_x == pytest.approx(0.1)
    assert measurement.gyro_y == pytest.approx(0.2)
    assert measurement.gyro_z == pytest.approx(0.3)
    assert measurement.accel_x == pytest.approx(1.0)
    assert measurement.accel_y == pytest.approx(-1.0)
    assert measurement.accel_z == pytest.approx(0.5)
    assert fusion._imu_timestamp == pytest.approx(1.0)


def test_update_imu_rejects_invalid_vector_size() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_imu(np.array([0.1, 0.2], dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0)


def test_update_imu_rejects_nan_inputs() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_imu(np.array([0.1, np.nan, 0.3], dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0)


def test_update_imu_rejects_inf_inputs() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_imu(np.array([0.1, 0.2, np.inf], dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0)


def test_update_imu_rejects_negative_timestamp() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_imu(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), -0.1)


def test_update_imu_rejects_non_monotonic_timestamp() -> None:
    fusion = SensorFusion()
    fusion.update_imu(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0)

    with pytest.raises(ValueError):
        fusion.update_imu(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0)


def test_update_gnss_stores_valid_measurement() -> None:
    fusion = SensorFusion()

    fusion.update_gnss(10.5, 20.25, 100.0, 2.0)

    measurement = fusion.get_latest_measurements()["gnss"]
    assert measurement is not None
    assert measurement.time == pytest.approx(2.0)
    assert measurement.latitude == pytest.approx(10.5)
    assert measurement.longitude == pytest.approx(20.25)
    assert measurement.altitude_m == pytest.approx(100.0)
    assert measurement.valid is True
    assert fusion._gnss_timestamp == pytest.approx(2.0)


def test_update_gnss_rejects_out_of_range_latitude() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_gnss(91.0, 0.0, 100.0, 1.0)


def test_update_gnss_rejects_out_of_range_longitude() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_gnss(0.0, 181.0, 100.0, 1.0)


def test_update_gnss_rejects_nan_inputs() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_gnss(np.nan, 0.0, 100.0, 1.0)


def test_update_gnss_rejects_inf_inputs() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_gnss(0.0, np.inf, 100.0, 1.0)


def test_update_gnss_rejects_negative_timestamp() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_gnss(0.0, 0.0, 100.0, -1.0)


def test_update_gnss_rejects_non_monotonic_timestamp() -> None:
    fusion = SensorFusion()
    fusion.update_gnss(0.0, 0.0, 100.0, 1.0)

    with pytest.raises(ValueError):
        fusion.update_gnss(0.0, 0.0, 100.0, 1.0)


def test_update_barometer_stores_valid_pressure() -> None:
    fusion = SensorFusion()

    fusion.update_barometer(101325.0, 3.0)

    measurement = fusion.get_latest_measurements()["barometer"]
    assert measurement is not None
    assert measurement.time == pytest.approx(3.0)
    assert measurement.pressure_pa == pytest.approx(101325.0)
    assert fusion._barometer_timestamp == pytest.approx(3.0)


def test_update_barometer_rejects_zero_pressure() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_barometer(0.0, 1.0)


def test_update_barometer_rejects_negative_pressure() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_barometer(-1.0, 1.0)


def test_update_barometer_rejects_nan_inputs() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_barometer(np.nan, 1.0)


def test_update_barometer_rejects_inf_inputs() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_barometer(np.inf, 1.0)


def test_update_barometer_rejects_negative_timestamp() -> None:
    fusion = SensorFusion()

    with pytest.raises(ValueError):
        fusion.update_barometer(101325.0, -1.0)


def test_update_barometer_rejects_non_monotonic_timestamp() -> None:
    fusion = SensorFusion()
    fusion.update_barometer(101325.0, 1.0)

    with pytest.raises(ValueError):
        fusion.update_barometer(101325.0, 1.0)


def test_get_latest_measurements_returns_all_three_measurements() -> None:
    fusion = SensorFusion()

    fusion.update_imu(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0)
    fusion.update_gnss(0.0, 0.0, 100.0, 2.0)
    fusion.update_barometer(101325.0, 3.0)

    measurements = fusion.get_latest_measurements()
    assert measurements["imu"] is not None
    assert measurements["gnss"] is not None
    assert measurements["barometer"] is not None


def test_missing_measurements_remain_none() -> None:
    fusion = SensorFusion()

    measurements = fusion.get_latest_measurements()
    assert measurements["imu"] is None
    assert measurements["gnss"] is None
    assert measurements["barometer"] is None


def test_updated_measurements_are_returned_correctly() -> None:
    fusion = SensorFusion()
    fusion.update_gnss(1.0, 2.0, 3.0, 1.0)

    measurements = fusion.get_latest_measurements()
    assert measurements["gnss"] is not None
    assert measurements["gnss"].latitude == pytest.approx(1.0)
    assert measurements["gnss"].longitude == pytest.approx(2.0)
    assert measurements["gnss"].altitude_m == pytest.approx(3.0)


def test_clear_removes_all_stored_packets() -> None:
    fusion = SensorFusion()
    fusion.update_imu(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0)
    fusion.update_gnss(0.0, 0.0, 100.0, 2.0)
    fusion.update_barometer(101325.0, 3.0)

    fusion.clear()

    measurements = fusion.get_latest_measurements()
    assert measurements["imu"] is None
    assert measurements["gnss"] is None
    assert measurements["barometer"] is None


def test_clear_resets_all_timestamps() -> None:
    fusion = SensorFusion()
    fusion.update_imu(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0)
    fusion.update_gnss(0.0, 0.0, 100.0, 2.0)
    fusion.update_barometer(101325.0, 3.0)

    fusion.clear()

    assert fusion._imu_timestamp is None
    assert fusion._gnss_timestamp is None
    assert fusion._barometer_timestamp is None


def test_validate_float64_inputs_are_preserved() -> None:
    fusion = SensorFusion()
    gyro = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    accel = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    fusion.update_imu(gyro, accel, np.float64(1.0))

    measurement = fusion.get_latest_measurements()["imu"]
    assert measurement is not None
    assert measurement.gyro_x == pytest.approx(0.1)
    assert measurement.accel_z == pytest.approx(3.0)
    assert measurement.time == pytest.approx(1.0)


def test_multiple_sensor_updates_do_not_interfere() -> None:
    fusion = SensorFusion()
    fusion.update_imu(np.array([0.1, 0.2, 0.3], dtype=np.float64), np.array([1.0, 2.0, 3.0], dtype=np.float64), 1.0)
    fusion.update_gnss(10.0, 20.0, 100.0, 2.0)
    fusion.update_barometer(101325.0, 3.0)

    imu_measurement = fusion.get_latest_measurements()["imu"]
    gnss_measurement = fusion.get_latest_measurements()["gnss"]
    barometer_measurement = fusion.get_latest_measurements()["barometer"]

    assert imu_measurement is not None
    assert gnss_measurement is not None
    assert barometer_measurement is not None
    assert imu_measurement.gyro_x == pytest.approx(0.1)
    assert gnss_measurement.latitude == pytest.approx(10.0)
    assert barometer_measurement.pressure_pa == pytest.approx(101325.0)


def test_updating_one_sensor_does_not_modify_the_others() -> None:
    fusion = SensorFusion()
    fusion.update_imu(np.array([0.1, 0.2, 0.3], dtype=np.float64), np.array([1.0, 2.0, 3.0], dtype=np.float64), 1.0)
    fusion.update_gnss(10.0, 20.0, 100.0, 2.0)

    measurements = fusion.get_latest_measurements()
    assert measurements["imu"] is not None
    assert measurements["gnss"] is not None
    assert measurements["barometer"] is None
