"""
gnc/interfaces.py
=================
Contract file for the GNC software stack built on top of RocketPy.

PURPOSE
-------
This file is the single source of truth for every data structure that
crosses a module boundary.  No engineer writes code that passes data
between modules without first defining (or referencing) a TypedDict /
dataclass / Protocol here.

RULE: If you change a type here, you own updating every module that
uses it AND bumping the version string below.  Open a PR against main
with both changes in the same commit.

VERSION
-------
GNC_INTERFACE_VERSION = "0.1.0"

OWNER
-----
Lead engineer.  All changes require lead approval before merge.

HOW TO USE
----------
    from gnc.interfaces import NavState, SensorPacket, ActuatorCommand, GNCConfig
    # then type-hint your function arguments with these types.

ROCKETPY CONTEXT
----------------
Inside a RocketPy _Controller callback the raw inputs are:

    time            : float               – simulation time in seconds
    sampling_rate   : float               – controller Hz
    state_vector    : list[float]         – [x,y,z, vx,vy,vz, e0,e1,e2,e3, wx,wy,wz]
    state_history   : list[list[float]]   – full history of state_vector
    observed_variables : list            – your custom logged data
    interactive_objects: list            – e.g. [air_brakes, gimbal]
    sensors         : list               – RocketPy sensor objects
    environment     : Environment        – wind, atmosphere, etc.

Sensor measurement attributes (read as sensor.measurement):
    Accelerometer  → tuple[float, float, float]   (ax, ay, az)  m/s²  body frame
    Gyroscope      → tuple[float, float, float]   (wx, wy, wz)  rad/s body frame
    GnssReceiver   → tuple[float, float, float]   (lat°, lon°, alt_m)
    Barometer      → float                         pressure in Pa  (ScalarSensor)

All types below translate these raw inputs into clean, versioned
contracts between the Nav, Guidance, and Control modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Version sentinel — increment on any breaking change
# ---------------------------------------------------------------------------

GNC_INTERFACE_VERSION: str = "0.1.0"


# ===========================================================================
# 1. SENSOR PACKET
#    Produced by: Nav engineer (sensor reading layer)
#    Consumed by: EKF predict / update steps
#    Source: raw sensor.measurement values from RocketPy sensors list
# ===========================================================================

@dataclass
class IMUPacket:
    """
    One time-stamped reading from an Inertial Measurement Unit.

    Populated directly from RocketPy sensor.measurement.

    Coordinate frame: BODY frame, right-hand rule.
        +x = nose direction
        +y = starboard
        +z = down (launch-rail convention)

    Units:
        accel_*  : m/s²
        gyro_*   : rad/s
        time     : seconds (simulation time)
    """

    time: float

    # Accelerometer (ax, ay, az) — includes gravity; subtract g in EKF predict
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0

    # Gyroscope (wx, wy, wz) — angular velocity in body frame
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

    @classmethod
    def from_sensors(
        cls,
        time: float,
        accelerometer,   # RocketPy Accelerometer instance
        gyroscope,       # RocketPy Gyroscope instance
    ) -> "IMUPacket":
        """
        Build an IMUPacket from live RocketPy sensor objects.

        Usage inside _Controller:
            imu = IMUPacket.from_sensors(time, sensors[0], sensors[1])
        """
        ax, ay, az = accelerometer.measurement  # tuple[float,float,float]
        wx, wy, wz = gyroscope.measurement      # tuple[float,float,float]
        return cls(
            time=time,
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=wx,  gyro_y=wy,  gyro_z=wz,
        )


@dataclass
class GNSSPacket:
    """
    One time-stamped GNSS fix.

    Populated from GnssReceiver.measurement = (lat, lon, altitude_m).

    Units:
        latitude   : decimal degrees  (positive = North)
        longitude  : decimal degrees  (positive = East)
        altitude_m : metres above MSL
        time       : seconds
    """

    time: float
    latitude:   float = 0.0
    longitude:  float = 0.0
    altitude_m: float = 0.0
    valid: bool = True          # set False if GNSS fix is lost

    @classmethod
    def from_sensor(cls, time: float, gnss_sensor) -> "GNSSPacket":
        lat, lon, alt = gnss_sensor.measurement
        return cls(time=time, latitude=lat, longitude=lon, altitude_m=alt)


@dataclass
class BaroPacket:
    """
    One barometric pressure reading.

    Barometer is a ScalarSensor: sensor.measurement → float (Pa).
    Convert to altitude using standard atmosphere in the Nav layer —
    do NOT convert here; keep raw Pa so Nav can apply its own model.

    Units:
        pressure_pa : Pascals
        time        : seconds
    """

    time: float
    pressure_pa: float = 101325.0   # sea-level default

    @classmethod
    def from_sensor(cls, time: float, baro_sensor) -> "BaroPacket":
        return cls(time=time, pressure_pa=float(baro_sensor.measurement))


@dataclass
class SensorPacket:
    """
    Aggregated sensor snapshot for one controller timestep.

    The Nav engineer builds this at the top of every _Controller call
    and passes it into the EKF.  Nothing downstream touches raw sensor
    objects — they only see SensorPacket.
    """

    time: float
    imu:  Optional[IMUPacket]  = None
    gnss: Optional[GNSSPacket] = None
    baro: Optional[BaroPacket] = None


# ===========================================================================
# 2. NAV STATE  (output of EKF / Navigation module)
#    Produced by: Nav engineer (EKF update step)
#    Consumed by: Controls engineer (all control loops)
#               : Guidance module (apogee estimation)
#               : Telemetry / logging
#
#    THIS IS THE MOST IMPORTANT CONTRACT IN THE FILE.
#    Nav and Controls must agree on every field before coding begins.
# ===========================================================================

@dataclass
class NavState:
    """
    Estimated navigation state output by the EKF.

    All quantities are in the INERTIAL (NED) frame unless noted.

    NED convention:
        +x = North
        +y = East
        +z = Down

    Quaternion convention: Hamilton, scalar-first  [q0, q1, q2, q3]
        Represents rotation from BODY frame to INERTIAL (NED) frame.
        q0² + q1² + q2² + q3² = 1  (enforced after each EKF update)

    Units:
        position    : metres      (from launch pad origin)
        velocity    : m/s
        attitude    : quaternion  (dimensionless)
        ang_vel     : rad/s       (body frame)
        altitude_m  : metres AGL  (derived: -pos_z relative to launch alt)
        accel_body  : m/s²        (body frame, gravity removed)
        valid       : bool        False if EKF has diverged → abort control
        time        : seconds
    """

    time: float = 0.0

    # Position in NED inertial frame (metres from launch pad)
    pos_north: float = 0.0
    pos_east:  float = 0.0
    pos_down:  float = 0.0          # negative = above launch pad

    # Velocity in NED inertial frame (m/s)
    vel_north: float = 0.0
    vel_east:  float = 0.0
    vel_down:  float = 0.0          # negative = upward

    # Attitude quaternion — body-to-NED  (Hamilton, scalar-first)
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0

    # Angular velocity in BODY frame (rad/s)
    ang_vel_x: float = 0.0
    ang_vel_y: float = 0.0
    ang_vel_z: float = 0.0

    # --- Derived convenience fields (computed by Nav, read-only downstream) ---

    # Altitude above ground level in metres (positive = above pad)
    altitude_m: float = 0.0

    # Total speed in m/s
    speed_ms: float = 0.0

    # Specific force in body frame (m/s²) — gravity subtracted
    accel_body_x: float = 0.0
    accel_body_y: float = 0.0
    accel_body_z: float = 0.0

    # --- EKF health ---
    valid: bool = True              # False → controls must enter safe mode
    covariance_trace: float = 0.0   # sum of diagonal of P matrix (health proxy)

    # --- Derived from raw RocketPy state_vector (used when EKF not yet ready) ---
    @classmethod
    def from_rocketpy_state(cls, time: float, sv: List[float]) -> "NavState":
        """
        Bootstrap NavState directly from the RocketPy state_vector.
        Use this ONLY for the first few timesteps before EKF converges,
        or during unit tests that bypass sensor simulation.

        RocketPy state_vector layout:
            sv[0:3]   = [x, y, z]            position (ENU, metres)
            sv[3:6]   = [vx, vy, vz]         velocity (ENU, m/s)
            sv[6:10]  = [e0, e1, e2, e3]     quaternion (RocketPy convention)
            sv[10:13] = [wx, wy, wz]         angular velocity body (rad/s)

        NOTE: RocketPy uses ENU (East-North-Up). This constructor converts
        to NED (North-East-Down) for consistency with the rest of the GNC stack.
        """
        x, y, z       = sv[0], sv[1], sv[2]
        vx, vy, vz    = sv[3], sv[4], sv[5]
        e0, e1, e2, e3 = sv[6], sv[7], sv[8], sv[9]
        wx, wy, wz    = sv[10], sv[11], sv[12]

        # ENU → NED:  north=y, east=x, down=-z
        speed = (vx**2 + vy**2 + vz**2) ** 0.5

        return cls(
            time=time,
            pos_north=y,   pos_east=x,   pos_down=-z,
            vel_north=vy,  vel_east=vx,  vel_down=-vz,
            q0=e0, q1=e1, q2=e2, q3=e3,
            ang_vel_x=wx, ang_vel_y=wy, ang_vel_z=wz,
            altitude_m=z,
            speed_ms=speed,
            valid=True,
        )


# ===========================================================================
# 3. GUIDANCE REFERENCE
#    Produced by: Guidance module (can be a simple precomputed profile)
#    Consumed by: Control loops as the setpoint / reference trajectory
# ===========================================================================

@dataclass
class GuidanceReference:
    """
    The desired state the control loops should track at this timestep.

    For Phase 1 (apogee targeting) most fields are unused — only
    target_altitude_m and flight_phase matter.

    For later phases (attitude control) target_* quaternion and
    angular rates become active.
    """

    time: float = 0.0

    # --- Apogee / altitude target ---
    target_altitude_m: float = 3000.0       # mission apogee target (AGL)
    estimated_apogee_m: float = 0.0         # EKF-based ballistic prediction

    # --- Attitude target (body-to-NED quaternion) ---
    target_q0: float = 1.0
    target_q1: float = 0.0
    target_q2: float = 0.0
    target_q3: float = 0.0

    # --- Angular rate target (rad/s, body frame) ---
    target_wx: float = 0.0
    target_wy: float = 0.0
    target_wz: float = 0.0

    # --- Flight phase (drives which control loops are active) ---
    flight_phase: str = "rail"
    # Valid values:
    #   "rail"      – on launch rail, no control
    #   "powered"   – motor burning, TVC active
    #   "coast"     – burnout to apogee, air-brakes active
    #   "descent"   – post-apogee, parachute logic
    #   "landed"    – on ground


# ===========================================================================
# 4. ACTUATOR COMMAND
#    Produced by: Control loops (controls engineer)
#    Consumed by: Actuator abstraction layer → RocketPy interactive objects
#
#    This is the OUTPUT of the GNC stack — what physically moves.
# ===========================================================================

@dataclass
class ActuatorCommand:
    """
    Commanded actuator positions for one controller timestep.

    The actuator layer reads this and calls the appropriate RocketPy
    interactive object methods.  Control loops ONLY write ActuatorCommand;
    they never touch RocketPy objects directly.

    All deflections follow right-hand rule in body frame.
    """

    time: float = 0.0

    # --- Air brakes ---
    # deployment_level in [0.0, 1.0]
    # 0.0 = fully retracted, 1.0 = fully deployed
    air_brake_deployment: float = 0.0

    # --- TVC gimbal (if fitted) ---
    # Deflection in radians.  Positive = nose tilts in +y / +z direction.
    gimbal_pitch_rad: float = 0.0   # pitch deflection
    gimbal_yaw_rad:   float = 0.0   # yaw deflection

    # --- Cold-gas RCS (future, placeholder) ---
    # Thrust fraction [0, 1] per thruster
    rcs_thrusters: List[float] = field(default_factory=list)

    # --- Safety ---
    safe_mode: bool = False
    # If True, actuator layer drives all outputs to safe/neutral positions.
    # Set by Nav when nav_state.valid == False.

    def clamp(self) -> "ActuatorCommand":
        """
        Enforce physical limits.  Call this before writing to RocketPy objects.
        Returns self (mutates in place) for chaining.
        """
        self.air_brake_deployment = max(0.0, min(1.0, self.air_brake_deployment))
        max_gimbal = 0.0872665  # 5 degrees in radians — adjust per vehicle
        self.gimbal_pitch_rad = max(-max_gimbal, min(max_gimbal, self.gimbal_pitch_rad))
        self.gimbal_yaw_rad   = max(-max_gimbal, min(max_gimbal, self.gimbal_yaw_rad))
        return self


def apply_actuator_command(cmd: ActuatorCommand, interactive_objects: list) -> None:
    """
    Write an ActuatorCommand to the RocketPy interactive objects.

    This function is the ONLY place in the codebase that touches
    RocketPy interactive objects from the control side.

    Expected interactive_objects order (set in Flight() constructor):
        interactive_objects[0] = AirBrakes instance
        interactive_objects[1] = GimbalActuator instance  (if fitted)

    Parameters
    ----------
    cmd : ActuatorCommand
        Clamped actuator command from the control loop.
    interactive_objects : list
        The list passed into the _Controller callback by RocketPy.
    """
    cmd.clamp()

    if cmd.safe_mode:
        # Drive everything to neutral
        if len(interactive_objects) > 0:
            interactive_objects[0].deployment_level = 0.0
        if len(interactive_objects) > 1:
            interactive_objects[1].pitch_deflection = 0.0
            interactive_objects[1].yaw_deflection   = 0.0
        return

    if len(interactive_objects) > 0:
        interactive_objects[0].deployment_level = cmd.air_brake_deployment

    if len(interactive_objects) > 1:
        interactive_objects[1].pitch_deflection = cmd.gimbal_pitch_rad
        interactive_objects[1].yaw_deflection   = cmd.gimbal_yaw_rad


# ===========================================================================
# 5. CONTROLLER LOG ENTRY
#    Produced by: _Controller callback (return value → observed_variables)
#    Consumed by: Post-flight analysis, ground station telemetry
# ===========================================================================

@dataclass
class ControllerLogEntry:
    """
    One row of logged GNC data appended to observed_variables each step.

    Return this (as a tuple or dict) from your _Controller function so
    RocketPy stores it in Flight.observed_variables for post-processing.

    Usage:
        return log_entry.to_tuple()
    """

    time:                  float
    nav_valid:             bool
    alt_estimated_m:       float
    alt_true_m:            float      # from state_vector[2] — for validation only
    apogee_estimated_m:    float
    air_brake_cmd:         float
    gimbal_pitch_cmd:      float
    gimbal_yaw_cmd:        float
    ekf_cov_trace:         float
    flight_phase:          str

    def to_tuple(self) -> tuple:
        """Serialize to tuple for RocketPy observed_variables list."""
        return (
            self.time,
            int(self.nav_valid),
            self.alt_estimated_m,
            self.alt_true_m,
            self.apogee_estimated_m,
            self.air_brake_cmd,
            self.gimbal_pitch_cmd,
            self.gimbal_yaw_cmd,
            self.ekf_cov_trace,
            self.flight_phase,
        )

    @classmethod
    def columns(cls) -> List[str]:
        """Column names for DataFrame / CSV export."""
        return [
            "time", "nav_valid", "alt_estimated_m", "alt_true_m",
            "apogee_estimated_m", "air_brake_cmd", "gimbal_pitch_cmd",
            "gimbal_yaw_cmd", "ekf_cov_trace", "flight_phase",
        ]


# ===========================================================================
# 6. GNC CONFIG
#    Produced by: Lead engineer (mission planning)
#    Consumed by: All modules at initialisation
#
#    All tunable parameters in one place.  No magic numbers in module code.
# ===========================================================================

@dataclass
class GNCConfig:
    """
    All tunable GNC parameters in one dataclass.

    Instantiate once at the top of your simulation script and pass to
    every GNC module constructor.  Changing a parameter here changes it
    everywhere — no hunting through source files.

    UNITS: SI throughout (metres, seconds, radians, kg).
    """

    # --- Mission parameters ---
    target_apogee_m: float = 3000.0         # desired apogee AGL (m)
    launch_altitude_msl: float = 0.0        # launch site altitude MSL (m)

    # --- Controller sampling ---
    gnc_rate_hz: float = 100.0              # _Controller sampling rate

    # --- EKF tuning (Nav engineer owns these) ---
    ekf_process_noise_pos: float   = 0.01  # Q matrix diagonal — position (m²)
    ekf_process_noise_vel: float   = 0.1   # Q — velocity (m/s)²
    ekf_process_noise_att: float   = 1e-4  # Q — attitude (rad²)
    ekf_process_noise_gyro: float  = 1e-5  # Q — gyro bias (rad/s)²

    # R matrix values derived from sensor.noise_density — overridden here
    # if you want to de-tune the filter for robustness
    ekf_r_accel: float  = 0.01    # accelerometer variance override (m/s²)²
    ekf_r_gyro:  float  = 1e-5    # gyroscope variance override  (rad/s)²
    ekf_r_baro:  float  = 25.0    # barometer altitude variance  (m²)
    ekf_r_gnss:  float  = 9.0     # GNSS position variance       (m²)

    # --- Apogee PID controller (Controls engineer owns these) ---
    apogee_pid_kp: float  = 0.8
    apogee_pid_ki: float  = 0.05
    apogee_pid_kd: float  = 0.2
    apogee_pid_max_integral: float = 0.5  # anti-windup clamp

    # --- Attitude LQR (Controls engineer owns these) ---
    # Q and R weight matrices stored as diagonal vectors for simplicity
    # Full matrices built inside lqr_controller.py
    lqr_q_att:     Tuple[float, ...] = (10.0, 10.0, 10.0)   # attitude error cost
    lqr_q_ang_vel: Tuple[float, ...] = (1.0, 1.0, 1.0)      # rate error cost
    lqr_r_gimbal:  Tuple[float, ...] = (100.0, 100.0)        # actuator effort cost

    # --- Actuator limits ---
    max_air_brake_deployment: float  = 1.0     # [0, 1]
    max_gimbal_deflection_rad: float = 0.0873  # 5 degrees

    # --- Phase transition thresholds ---
    rail_clear_velocity_ms: float    = 20.0    # speed above which rail phase ends
    burnout_detection_accel: float   = 0.5     # net accel threshold for burnout (m/s²)
    apogee_detection_vel_ms: float   = 2.0     # vertical velocity at apogee (m/s)

    # --- Safety ---
    ekf_divergence_cov_limit: float  = 1e6     # covariance trace above = invalid


# ===========================================================================
# 7. MODULE PROTOCOLS  (what each module must implement)
#    These are the method signatures Nav and Controls engineers must match.
#    Think of these as the "interface" in the Java/C# sense.
# ===========================================================================

class NavigationModuleProtocol:
    """
    Every navigation module (EKF or otherwise) must implement this API.
    Nav engineer: subclass this or duck-type it — both are fine.

    The _Controller callback calls nav.update(sensor_packet) then reads
    nav.state to get the current NavState.
    """

    @property
    def state(self) -> NavState:
        """Current best estimate of vehicle state."""
        raise NotImplementedError

    def update(self, sensor_packet: SensorPacket) -> NavState:
        """
        Run one EKF predict + update cycle.

        Parameters
        ----------
        sensor_packet : SensorPacket
            All sensor readings for this timestep.

        Returns
        -------
        NavState
            Updated state estimate.  Also stored in self.state.
        """
        raise NotImplementedError

    def reset(self, initial_state: NavState) -> None:
        """Reset filter to a known state (used at rail-clear event)."""
        raise NotImplementedError


class ControlModuleProtocol:
    """
    Every control module must implement this API.

    The _Controller callback calls controller.compute(nav_state, guidance)
    and gets back an ActuatorCommand.
    """

    def compute(
        self,
        nav_state: NavState,
        guidance: GuidanceReference,
        dt: float,
    ) -> ActuatorCommand:
        """
        Compute actuator commands for this timestep.

        Parameters
        ----------
        nav_state : NavState
            Current vehicle state from the Nav module.
        guidance : GuidanceReference
            Target / reference from the Guidance module.
        dt : float
            Time since last call in seconds (= 1 / gnc_rate_hz nominally).

        Returns
        -------
        ActuatorCommand
            Desired actuator positions.  NOT yet clamped — call
            apply_actuator_command() which clamps before writing.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset integrators / state on phase transition."""
        raise NotImplementedError


class GuidanceModuleProtocol:
    """
    Guidance module computes the reference trajectory / setpoints.
    """

    def compute_reference(
        self,
        nav_state: NavState,
        config: GNCConfig,
    ) -> GuidanceReference:
        """
        Return the current guidance reference given vehicle state.

        Parameters
        ----------
        nav_state : NavState
        config : GNCConfig

        Returns
        -------
        GuidanceReference
        """
        raise NotImplementedError


# ===========================================================================
# 8. TOP-LEVEL CONTROLLER FUNCTION TEMPLATE
#    Copy this into your controller module.  Fill in the blanks.
#    Do NOT modify the function signature — it matches RocketPy's _Controller.
# ===========================================================================

def gnc_controller_template(
    time: float,
    sampling_rate: float,
    state_vector: List[float],
    state_history: List[List[float]],
    observed_variables: list,
    interactive_objects: list,
    sensors: list,
    environment,
) -> tuple:
    """
    Master GNC controller function — plug into RocketPy _Controller.

    This is a template.  The real implementation lives in gnc/controller.py.
    Each sub-module (nav, guidance, control) is instantiated once outside
    this function (as closures or module-level singletons) and called here.

    Returns
    -------
    tuple
        ControllerLogEntry.to_tuple() — appended to observed_variables by RocketPy.
    """
    dt = 1.0 / sampling_rate

    # --- 1. Build sensor packet from RocketPy sensor objects ---
    # Nav engineer fills this in — they own sensor indexing
    sensor_packet = SensorPacket(time=time)
    # sensor_packet.imu  = IMUPacket.from_sensors(time, sensors[0], sensors[1])
    # sensor_packet.gnss = GNSSPacket.from_sensor(time, sensors[2])
    # sensor_packet.baro = BaroPacket.from_sensor(time, sensors[3])

    # --- 2. Run EKF ---
    # nav_state = nav_module.update(sensor_packet)
    nav_state = NavState.from_rocketpy_state(time, state_vector)  # placeholder

    # --- 3. Compute guidance reference ---
    # guidance = guidance_module.compute_reference(nav_state, config)
    guidance = GuidanceReference(time=time)  # placeholder

    # --- 4. Run control laws ---
    # cmd = control_module.compute(nav_state, guidance, dt)
    cmd = ActuatorCommand(time=time)  # placeholder

    # --- 5. Write to actuators ---
    apply_actuator_command(cmd, interactive_objects)

    # --- 6. Log and return ---
    log = ControllerLogEntry(
        time=time,
        nav_valid=nav_state.valid,
        alt_estimated_m=nav_state.altitude_m,
        alt_true_m=state_vector[2],
        apogee_estimated_m=guidance.estimated_apogee_m,
        air_brake_cmd=cmd.air_brake_deployment,
        gimbal_pitch_cmd=cmd.gimbal_pitch_rad,
        gimbal_yaw_cmd=cmd.gimbal_yaw_rad,
        ekf_cov_trace=nav_state.covariance_trace,
        flight_phase=guidance.flight_phase,
    )
    return log.to_tuple()
