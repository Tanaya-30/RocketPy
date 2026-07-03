"""
Linear Quadratic Regulator (LQR) for RocketPy GNC.

This module implements a continuous-time Linear Quadratic Regulator
used for attitude stabilization and thrust vector control.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from rocketpy.gnc.interfaces import (
    GNCConfig,
)


class LQRController:
    """
    Continuous-time Linear Quadratic Regulator.

    The controller computes the optimal state feedback

        u = -Kx

    where K is the feedback gain matrix.
    """

    def __init__(
        self,
        config: GNCConfig,
    ) -> None:
        """
        Initialize the LQR controller.
        """

        if not isinstance(
            config,
            GNCConfig,
        ):
            raise TypeError(
                "config must be a GNCConfig."
            )

        self._config = config

        self._q = np.diag(
            [
                config.lqr_q_att,
                config.lqr_q_att,
                config.lqr_q_att,
                config.lqr_q_ang_vel,
                config.lqr_q_ang_vel,
                config.lqr_q_ang_vel,
            ]
        )

        self._r = np.diag(
            [
                config.lqr_r_gimbal,
                config.lqr_r_gimbal,
            ]
        )

        self._gain: NDArray[np.float64] | None = None

    @property
    def gain(
        self,
    ) -> NDArray[np.float64] | None:
        """
        Return the current LQR gain matrix.
        """

        if self._gain is None:
            return None

        return self._gain.copy()
    
    def set_gain(
        self,
        gain: NDArray[np.float64],
    ) -> None:
        """
        Set the LQR feedback gain matrix.

        Parameters
        ----------
        gain : ndarray
            Feedback gain matrix.
        """

        gain = np.asarray(
            gain,
            dtype=np.float64,
        )

        if gain.shape != (2, 6):
            raise ValueError(
                "gain must have shape (2, 6)."
            )

        if not np.all(
            np.isfinite(gain)
        ):
            raise ValueError(
                "gain contains non-finite values."
            )

        self._gain = gain.copy()

    def compute(
        self,
        state_error: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the optimal control command.

        Parameters
        ----------
        state_error : ndarray
            State error vector.

        Returns
        -------
        ndarray
            Control command.
        """

        if self._gain is None:
            raise RuntimeError(
                "LQR gain has not been initialized."
            )

        state_error = np.asarray(
            state_error,
            dtype=np.float64,
        )

        if state_error.shape != (6,):
            raise ValueError(
                "state_error must have shape (6,)."
            )

        if not np.all(
            np.isfinite(state_error)
        ):
            raise ValueError(
                "state_error contains non-finite values."
            )

        control = -self._gain @ state_error

        return control.astype(
            np.float64,
        )
    
    @property
    def config(
        self,
    ) -> GNCConfig:
        """
        Return the controller configuration.
        """

        return self._config

    @property
    def q(
        self,
    ) -> NDArray[np.float64]:
        """
        Return the state weighting matrix.
        """

        return self._q.copy()

    @property
    def r(
        self,
    ) -> NDArray[np.float64]:
        """
        Return the control weighting matrix.
        """

        return self._r.copy()

    def reset(
        self,
    ) -> None:
        """
        Reset the controller.
        """

        self._gain = None

    def copy(
        self,
    ) -> "LQRController":
        """
        Return a deep copy of the controller.
        """

        controller = LQRController(
            self._config,
        )

        if self._gain is not None:
            controller._gain = (
                self._gain.copy()
            )

        return controller

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        gain = (
            "set"
            if self._gain is not None
            else "unset"
        )

        return (
            f"{self.__class__.__name__}("
            f"gain={gain}"
            ")"
        )
    
__all__ = [
    "LQRController",
]