"""
Core Guidance Law implementation for RocketPy GNC.

The GuidanceLaw coordinates all guidance submodules and produces the
GuidanceReference consumed by the control subsystem.
"""

from __future__ import annotations

from rocketpy.gnc.guidance.dispersion_guidance import (
    DispersionGuidance,
)
from rocketpy.gnc.guidance.landing_guidance import (
    LandingGuidance,
)
from rocketpy.gnc.guidance.phase_manager import (
    FlightPhase,
    PhaseManager,
)
from rocketpy.gnc.guidance.pitch_program import (
    PitchProgram,
)
from rocketpy.gnc.guidance.targeting import (
    MissionTarget,
)
from rocketpy.gnc.guidance.trajectory import (
    ReferenceTrajectory,
)
from rocketpy.gnc.guidance.trajectory_tracker import (
    TrajectoryTracker,
)
from rocketpy.gnc.interfaces import (
    GNCConfig,
    GuidanceModuleProtocol,
    GuidanceReference,
    NavState,
)


class GuidanceLaw(GuidanceModuleProtocol):
    """
    Master guidance coordinator.

    This class combines the individual guidance modules into a single
    GuidanceReference used by the flight controller.
    """

    def __init__(
        self,
        config: GNCConfig,
    ) -> None:
        """
        Initialize the guidance system.

        The internal guidance modules are constructed
        and initialized with their default state.
        """

        self._target: MissionTarget | None = None

        self._phase_manager = PhaseManager()

        self._pitch_program = PitchProgram()

        self._trajectory_tracker = TrajectoryTracker()

        self._landing_guidance = LandingGuidance()

        self._dispersion_guidance = DispersionGuidance()

    @property
    def target(
        self,
    ) -> MissionTarget | None:
        """
        Return the current mission target.

        The returned mission target is retrieved from
        the internally stored guidance target.
        """

        return self._target

    def set_target(
        self,
        target: MissionTarget,
    ) -> None:
        """
        Set the mission target.

        The supplied mission target is stored and
        propagated to the landing and dispersion
        guidance modules.
        """

        self._target = target

        self._landing_guidance.set_target(target)

        self._dispersion_guidance.set_target(target)

    def clear_target(
        self,
    ) -> None:
        """
        Remove the current mission target.
        """

        self._target = None

        self._landing_guidance.clear()

        self._dispersion_guidance.clear()

    def set_reference_trajectory(
        self,
        trajectory: ReferenceTrajectory,
    ) -> None:
        """
        Set the active reference trajectory.

        The supplied trajectory is stored by the
        trajectory tracker.
        """

        self._trajectory_tracker.set_trajectory(trajectory)

    def clear_reference_trajectory(
        self,
    ) -> None:
        """
        Clear the active reference trajectory.

        The internally stored trajectory is removed
        from the trajectory tracker.
        """

        self._trajectory_tracker.clear()

    @property
    def phase(
        self,
    ) -> FlightPhase:
        """
        Return the current flight phase.

        The returned phase is retrieved from the
        phase manager.
        """

        return self._phase_manager.phase

    def compute_reference(
        self,
        nav_state: NavState,
        config: GNCConfig,
    ) -> GuidanceReference:
        """
        Compute the guidance reference.

        The returned guidance reference is computed
        from the current navigation state, mission
        configuration and active guidance modules.

        Parameters
        ----------
        nav_state : NavState
            Current navigation state.

        config : GNCConfig
            Global GNC configuration.

        Returns
        -------
        GuidanceReference
            Guidance reference computed for the
            current flight state.
        """

        phase = self._phase_manager.update(
            nav_state,
            config,
        )

        reference = GuidanceReference(
            time=nav_state.time,
        )

        reference.flight_phase = phase.value

        if self._target is None:
            reference.target_altitude_m = config.target_apogee_m
        else:
            reference.target_altitude_m = self._target.target_apogee

        if self._trajectory_tracker.has_trajectory():
            waypoint = self._trajectory_tracker.reference_at_time(
                nav_state.time,
            )

            reference.target_q0 = waypoint.attitude.w

            reference.target_q1 = waypoint.attitude.x

            reference.target_q2 = waypoint.attitude.y

            reference.target_q3 = waypoint.attitude.z

            reference.target_wx = float(waypoint.angular_velocity[0])

            reference.target_wy = float(waypoint.angular_velocity[1])

            reference.target_wz = float(waypoint.angular_velocity[2])

        elif len(self._pitch_program) > 0:
            quaternion = self._pitch_program.quaternion_at_time(
                nav_state.time,
            )

            reference.target_q0 = quaternion.w

            reference.target_q1 = quaternion.x

            reference.target_q2 = quaternion.y

            reference.target_q3 = quaternion.z

        reference.estimated_apogee_m = nav_state.altitude_m

        if phase is FlightPhase.DESCENT:
            landing_reference = self._landing_guidance.compute_reference(
                nav_state,
            )

            reference.target_altitude_m = landing_reference.target_altitude_m
        return reference

    def reset(
        self,
    ) -> None:
        """
        Reset the guidance law.
        """

        self._target = None

        self._phase_manager.reset()

        self._trajectory_tracker.reset()

        self._landing_guidance.reset()

        self._dispersion_guidance.reset()

    @property
    def phase_manager(
        self,
    ) -> PhaseManager:
        """
        Return the phase manager.

        The returned phase manager is retrieved from
        the internally stored guidance modules.
        """

        return self._phase_manager

    @property
    def pitch_program(
        self,
    ) -> PitchProgram:
        """
        Return the pitch program.

        The returned pitch program is retrieved from
        the internally stored guidance modules.
        """

        return self._pitch_program

    @property
    def trajectory_tracker(
        self,
    ) -> TrajectoryTracker:
        """
        Return the trajectory tracker.

        The returned trajectory tracker is retrieved
        from the internally stored guidance modules.
        """

        return self._trajectory_tracker

    @property
    def landing_guidance(
        self,
    ) -> LandingGuidance:
        """
        Return the landing guidance module.

        The returned landing guidance module is
        retrieved from the internally stored guidance
        modules.
        """

        return self._landing_guidance

    @property
    def dispersion_guidance(
        self,
    ) -> DispersionGuidance:
        """
        Return the dispersion guidance module.

        The returned dispersion guidance module is
        retrieved from the internally stored guidance
        modules.
        """

        return self._dispersion_guidance

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"phase={self.phase.value}, "
            f"target={'set' if self._target else 'none'}, "
            f"trajectory={self._trajectory_tracker.has_trajectory()}"
            ")"
        )


__all__ = [
    "GuidanceLaw",
]
