"""Unit tests for SensorPipeline."""

import numpy as np
import pytest

from rocketpy.gnc.navigation.sensor_packet import SensorPacket
from rocketpy.gnc.navigation.sensor_pipeline import SensorPipeline


def make_packet(timestamp: float = 0.0) -> SensorPacket:
    """Create a valid sensor packet."""

    return SensorPacket(
        timestamp=timestamp,
        accelerometer=np.array([0.1, 0.0, 9.81]),
        gyroscope=np.array([0.01, 0.02, 0.03]),
        gps_position=np.array([18.5204, 73.8567, 560.0]),
        barometric_altitude=558.4,
    )


def test_pipeline_initialization():
    """Pipeline initializes correctly."""

    pipeline = SensorPipeline()

    state = pipeline.get_state()

    assert state.shape == (13,)
    assert np.isclose(state[6], 1.0)


def test_single_update():
    """Pipeline accepts one sensor packet."""

    pipeline = SensorPipeline()

    state = pipeline.update(make_packet())

    assert state.shape == (13,)
    assert np.all(np.isfinite(state))


def test_multiple_updates():
    """Pipeline processes multiple packets."""

    pipeline = SensorPipeline()

    for i in range(10):
        state = pipeline.update(
            make_packet(timestamp=i * 0.1)
        )

    assert state.shape == (13,)
    assert np.all(np.isfinite(state))


def test_reset():
    """Pipeline reset restores default state."""

    pipeline = SensorPipeline()

    pipeline.update(make_packet())

    pipeline.reset()

    state = pipeline.get_state()

    assert np.allclose(state[0:3], np.zeros(3))
    assert np.allclose(state[3:6], np.zeros(3))
    assert np.isclose(state[6], 1.0)


def test_invalid_packet():
    """Pipeline rejects invalid packet types."""

    pipeline = SensorPipeline()

    with pytest.raises(TypeError):
        pipeline.update(None)


def test_timestamp_order():
    """Timestamps must increase."""

    pipeline = SensorPipeline()

    pipeline.update(make_packet(0.5))

    with pytest.raises(ValueError):
        pipeline.update(make_packet(0.4))


def test_pipeline_state_dimension():
    """State vector has expected dimension."""

    pipeline = SensorPipeline()

    state = pipeline.get_state()

    assert len(state) == 13