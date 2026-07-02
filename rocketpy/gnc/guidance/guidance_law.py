"""
Core Guidance Law implementation for RocketPy GNC.

Implements the GuidanceModuleProtocol defined in gnc.interfaces.
The GuidanceLaw converts the current navigation state into the
desired GuidanceReference for the active flight phase.
"""

from __future__ import annotations

from rocketpy.gnc.guidance.targeting import MissionTarget
from rocketpy.gnc.guidance.trajectory import ReferenceTrajectory
from rocketpy.gnc.interfaces import (
    GNCConfig,
    GuidanceModuleProtocol,
    GuidanceReference,
    NavState,
)


class GuidanceLaw(GuidanceModuleProtocol):
    """
    Default RocketPy guidance implementation.

    This class owns the high-level flight guidance logic.
    Individual guidance algorithms (pitch program, apogee guidance,
    landing guidance, etc.) will be delegated to dedicated modules.
    """

    def __init__(
        self,
        config: GNCConfig,
    ) -> None:
        """
        Initialize the guidance law.

        Parameters
        ----------
        config : GNCConfig
            Global GNC configuration.
        """

        self._config = config

        self._target: MissionTarget | None = None

        self._trajectory: ReferenceTrajectory | None = None

    def set_target(
        self,
        target: MissionTarget,
    ) -> None:
        """
        Set the active mission target.
        """

        self._target = target

    def clear_target(
        self,
    ) -> None:
        """
        Remove the active mission target.
        """

        self._target = None

    def set_reference_trajectory(
        self,
        trajectory: ReferenceTrajectory,
    ) -> None:
        """
        Set the reference trajectory.
        """

        self._trajectory = trajectory

    def clear_reference_trajectory(
        self,
    ) -> None:
        """
        Remove the current trajectory.
        """

        self._trajectory = None

    def compute_reference(
        self,
        nav_state: NavState,
        config: GNCConfig,
    ) -> GuidanceReference:
        """
        Compute the guidance reference for the current timestep.

        Parameters
        ----------
        nav_state : NavState
            Current navigation state.
        config : GNCConfig
            Global GNC configuration.

        Returns
        -------
        GuidanceReference
            Desired guidance reference.
        """

        reference = GuidanceReference(
            time=nav_state.time,
        )

        if self._target is None:
            reference.target_altitude_m = (
                config.target_apogee_m
            )
        else:
            reference.target_altitude_m = (
                self._target.target_apogee
            )

        reference.estimated_apogee_m = max(
            nav_state.altitude_m,
            self._estimate_apogee(
                nav_state,
            ),
        )

        reference.flight_phase = self._determine_flight_phase(
            nav_state,
            config,
        )

        return reference
    
    def _estimate_apogee(
        self,
        nav_state: NavState,
    ) -> float:
        """
        Estimate the ballistic apogee.

        Parameters
        ----------
        nav_state : NavState
            Current navigation state.

        Returns
        -------
        float
            Estimated apogee above ground level.
        """

        gravity = 9.80665

        vertical_velocity = -nav_state.vel_down

        if vertical_velocity <= 0.0:
            return nav_state.altitude_m

        return (
            nav_state.altitude_m
            + (vertical_velocity ** 2)
            / (2.0 * gravity)
        )

    def _determine_flight_phase(
        self,
        nav_state: NavState,
        config: GNCConfig,
    ) -> str:
        """
        Determine the current flight phase.

        Parameters
        ----------
        nav_state : NavState
            Current navigation state.

        config : GNCConfig
            Global GNC configuration.

        Returns
        -------
        str
            Flight phase.
        """

        upward_velocity = -nav_state.vel_down

        if nav_state.speed_ms < config.rail_clear_velocity_ms:
            return "rail"

        if (
            nav_state.accel_body_z
            > config.burnout_detection_accel
        ):
            return "powered"

        if upward_velocity > config.apogee_detection_vel_ms:
            return "coast"

        if nav_state.altitude_m > 1.0:
            return "descent"

        return "landed"
    
    def reset(
        self,
    ) -> None:
        """
        Reset the guidance law.
        """

        self._target = None
        self._trajectory = None

    @property
    def target(
        self,
    ) -> MissionTarget | None:
        """
        Return the active mission target.
        """

        return self._target

    @property
    def reference_trajectory(
        self,
    ) -> ReferenceTrajectory | None:
        """
        Return the active reference trajectory.
        """

        return self._trajectory
    
