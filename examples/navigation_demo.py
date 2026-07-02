"""
Navigation pipeline demonstration.

This example generates random sensor measurements, creates SensorPacket
objects, processes them through the SensorPipeline, and prints the
estimated navigation state.
"""

from __future__ import annotations

import time

import numpy as np

from rocketpy.gnc.navigation.sensor_packet import SensorPacket
from rocketpy.gnc.navigation.sensor_pipeline import SensorPipeline


def generate_random_packet(timestamp: float) -> SensorPacket:
    """
    Generate a random sensor packet.

    Parameters
    ----------
    timestamp : float
        Simulation timestamp.

    Returns
    -------
    SensorPacket
        Randomly generated sensor packet.
    """

    accelerometer = np.random.normal(
        loc=0.0,
        scale=0.2,
        size=3,
    )

    gyroscope = np.random.normal(
        loc=0.0,
        scale=0.02,
        size=3,
    )

    gps_position = np.array(
        [
            np.random.uniform(-90.0, 90.0),
            np.random.uniform(-180.0, 180.0),
            np.random.uniform(0.0, 500.0),
        ],
        dtype=np.float64,
    )

    barometric_altitude = np.random.uniform(
        0.0,
        500.0,
    )

    return SensorPacket(
        timestamp=timestamp,
        accelerometer=accelerometer,
        gyroscope=gyroscope,
        gps_position=gps_position,
        barometric_altitude=barometric_altitude,
    )


def main() -> None:
    """Run the navigation pipeline demonstration."""

    pipeline = SensorPipeline()

    timestamp = 0.0
    dt = 0.1

    print("=" * 70)
    print("RocketPy Navigation Pipeline Demo")
    print("=" * 70)

    for step in range(10):

        packet = generate_random_packet(timestamp)

        state = pipeline.update(packet)

        print(f"\nStep {step + 1}")
        print(f"Time      : {timestamp:.2f} s")
        print(f"Accel     : {packet.accelerometer}")
        print(f"Gyro      : {packet.gyroscope}")
        print(f"GPS       : {packet.gps_position}")
        print(f"Barometer : {packet.barometric_altitude:.2f} m")

        print("\nEstimated Navigation State")
        print("--------------------------")
        print(f"Position      : {state[0:3]}")
        print(f"Velocity      : {state[3:6]}")
        print(f"Quaternion    : {state[6:10]}")
        print(f"Angular Rate  : {state[10:13]}")

        timestamp += dt
        time.sleep(0.2)


if __name__ == "__main__":
    main()