"""
Unit tests for lqr.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocketpy.gnc.control.lqr import (
    LQRController,
)
from rocketpy.gnc.interfaces import (
    GNCConfig,
)


def create_config() -> GNCConfig:
    config = GNCConfig()

    config.lqr_q_att = 10.0
    config.lqr_q_ang_vel = 2.0
    config.lqr_r_gimbal = 1.0

    return config


def create_gain() -> np.ndarray:
    return np.ones(
        (
            2,
            6,
        ),
        dtype=np.float64,
    )


def test_controller_creation() -> None:

    controller = LQRController(
        create_config()
    )

    assert controller.config is not None


def test_invalid_config() -> None:

    with pytest.raises(TypeError):

        LQRController(
            None,
        )


def test_q_matrix_shape() -> None:

    controller = LQRController(
        create_config()
    )

    assert controller.q.shape == (
        6,
        6,
    )


def test_r_matrix_shape() -> None:

    controller = LQRController(
        create_config()
    )

    assert controller.r.shape == (
        2,
        2,
    )


def test_set_gain() -> None:

    controller = LQRController(
        create_config()
    )

    gain = create_gain()

    controller.set_gain(
        gain,
    )

    assert np.array_equal(
        controller.gain,
        gain,
    )


def test_invalid_gain_shape() -> None:

    controller = LQRController(
        create_config()
    )

    with pytest.raises(ValueError):

        controller.set_gain(
            np.ones(
                (
                    3,
                    6,
                )
            )
        )


def test_compute() -> None:

    controller = LQRController(
        create_config()
    )

    controller.set_gain(
        create_gain()
    )

    state = np.ones(
        6,
        dtype=np.float64,
    )

    control = controller.compute(
        state,
    )

    assert control.shape == (
        2,
    )


def test_compute_without_gain() -> None:

    controller = LQRController(
        create_config()
    )

    with pytest.raises(RuntimeError):

        controller.compute(
            np.zeros(
                6,
            )
        )


def test_invalid_state_shape() -> None:

    controller = LQRController(
        create_config()
    )

    controller.set_gain(
        create_gain()
    )

    with pytest.raises(ValueError):

        controller.compute(
            np.zeros(
                5,
            )
        )


def test_reset() -> None:

    controller = LQRController(
        create_config()
    )

    controller.set_gain(
        create_gain()
    )

    controller.reset()

    assert controller.gain is None


def test_copy() -> None:

    controller = LQRController(
        create_config()
    )

    controller.set_gain(
        create_gain()
    )

    copied = controller.copy()

    assert copied is not controller

    assert np.array_equal(
        copied.gain,
        controller.gain,
    )


def test_repr() -> None:

    controller = LQRController(
        create_config()
    )

    assert "LQRController" in repr(
        controller
    )