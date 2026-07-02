"""Unit tests for SensorPacket."""

import numpy as np
import pytest

from rocketpy.gnc.navigation.sensor_packet import SensorPacket


def make_packet() -> SensorPacket:
    """Create a valid sensor packet."""

    return SensorPacket(
        timestamp=0.0,
        accelerometer=np.array([0.1, 0.2, 9.81]),
        gyroscope=np.array([0.01, 0.02, 0.03]),
        gps_position=np.array([18.5204, 73.8567, 560.0]),
        barometric_altitude=558.4,
    )


def test_sensor_packet_creation():
    """Verify packet creation."""

    packet = make_packet()

    assert packet.timestamp == 0.0
    assert np.array_equal(
        packet.accelerometer,
        np.array([0.1, 0.2, 9.81]),
    )
    assert np.array_equal(
        packet.gyroscope,
        np.array([0.01, 0.02, 0.03]),
    )
    assert np.array_equal(
        packet.gps_position,
        np.array([18.5204, 73.8567, 560.0]),
    )
    assert packet.barometric_altitude == 558.4


def test_negative_timestamp():
    """Timestamp must be non-negative."""

    with pytest.raises(ValueError):
        SensorPacket(
            timestamp=-1.0,
            accelerometer=np.zeros(3),
            gyroscope=np.zeros(3),
            gps_position=np.zeros(3),
            barometric_altitude=0.0,
        )


def test_invalid_accelerometer_shape():
    """Accelerometer must contain three values."""

    with pytest.raises(ValueError):
        SensorPacket(
            timestamp=0.0,
            accelerometer=np.zeros(2),
            gyroscope=np.zeros(3),
            gps_position=np.zeros(3),
            barometric_altitude=0.0,
        )


def test_invalid_gyroscope_shape():
    """Gyroscope must contain three values."""

    with pytest.raises(ValueError):
        SensorPacket(
            timestamp=0.0,
            accelerometer=np.zeros(3),
            gyroscope=np.zeros(4),
            gps_position=np.zeros(3),
            barometric_altitude=0.0,
        )


def test_invalid_gps_shape():
    """GPS position must contain three values."""

    with pytest.raises(ValueError):
        SensorPacket(
            timestamp=0.0,
            accelerometer=np.zeros(3),
            gyroscope=np.zeros(3),
            gps_position=np.zeros(2),
            barometric_altitude=0.0,
        )


def test_non_finite_barometer():
    """Barometric altitude must be finite."""

    with pytest.raises(ValueError):
        SensorPacket(
            timestamp=0.0,
            accelerometer=np.zeros(3),
            gyroscope=np.zeros(3),
            gps_position=np.zeros(3),
            barometric_altitude=np.nan,
        )


def test_packet_is_immutable():
    """SensorPacket should be immutable."""

    packet = make_packet()

    with pytest.raises(Exception):
        packet.timestamp = 5.0