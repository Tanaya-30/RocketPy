"""Quaternion mathematics for RocketGNC.

This module provides a numerically stable quaternion implementation for
attitude representation and rotation operations in aerospace navigation
software. The implementation uses the Hamilton convention with storage order
[w, x, y, z], active rotations, and a right-handed coordinate system. Euler
angles use the aerospace ZYX convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12
ZERO_NORM_TOL = 1e-15
ORTHOGONALITY_TOL = 1e-10
ANGLE_TOL = 1e-14


class Quaternion:
    """
    Represent a quaternion stored as [w, x, y, z].

    Supports attitude operations for navigation and frame transformations.
    """

    def __init__(
        self,
        w: float | np.floating[Any],
        x: float | np.floating[Any],
        y: float | np.floating[Any],
        z: float | np.floating[Any],
    ) -> None:
        """
        Initialize a quaternion from scalar components stored in self.q.
        """
        self.q: NDArray[np.float64] = np.array(
            [np.float64(w), np.float64(x), np.float64(y), np.float64(z)],
            dtype=np.float64,
        )

    @property
    def w(self) -> np.float64:
        """
        Return the scalar component.

        Retrieved from the internally stored quaternion
        component ``self.q[0]``.
        """
        return np.float64(self.q[0])

    @property
    def x(self) -> np.float64:
        """
        Return the x component.

        Retrieved from the internally stored quaternion
        component ``self.q[1]``.
        """
        return np.float64(self.q[1])

    @property
    def y(self) -> np.float64:
        """
        Return the y component.

        Retrieved from the internally stored quaternion
        component ``self.q[2]``.
        """
        return np.float64(self.q[2])

    @property
    def z(self) -> np.float64:
        """
        Return the z component.

        Retrieved from the internally stored quaternion
        component ``self.q[3]``.
        """
        return np.float64(self.q[3])

    @classmethod
    def identity(cls) -> "Quaternion":
        """
        Return the identity quaternion.

        Constructed from the Hamilton components
        [1, 0, 0, 0], representing zero rotation.
        """
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_numpy(
        cls,
        q: NDArray[np.float64] | list[float] | tuple[float, ...],
    ) -> "Quaternion":
        """
        Construct a quaternion from an array-like object.

        The quaternion components are copied from the
        input sequence in Hamilton order [w, x, y, z].
        """
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (4,):
            raise ValueError("Quaternion input must be a 1D array-like of length 4.")
        return cls(values[0], values[1], values[2], values[3])

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> "Quaternion":
        """
        Construct a quaternion from Euler angles.

        Computed from the supplied roll, pitch and yaw
        angles using the aerospace ZYX convention.
        """
        roll_half = np.float64(0.5 * roll)
        pitch_half = np.float64(0.5 * pitch)
        yaw_half = np.float64(0.5 * yaw)

        cr = np.cos(roll_half)
        sr = np.sin(roll_half)
        cp = np.cos(pitch_half)
        sp = np.sin(pitch_half)
        cy = np.cos(yaw_half)
        sy = np.sin(yaw_half)

        return cls(
            np.float64(cy * cp * cr + sy * sp * sr),
            np.float64(cy * cp * sr - sy * sp * cr),
            np.float64(cy * sp * cr + sy * cp * sr),
            np.float64(sy * cp * cr - cy * sp * sr),
        )

    @classmethod
    def from_rotation_matrix(cls, R: NDArray[np.float64]) -> "Quaternion":
        """
        Construct a quaternion from a rotation matrix.

        Computed from the supplied 3×3 rotation matrix
        after validating its orthogonality.
        """
        rotation = np.asarray(R, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("Rotation matrix must have shape (3, 3).")

        identity = np.eye(3, dtype=np.float64)
        if not np.allclose(
            rotation.T @ rotation, identity, atol=ORTHOGONALITY_TOL, rtol=0.0
        ):
            raise ValueError("Rotation matrix is not orthogonal within tolerance.")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=ORTHOGONALITY_TOL):
            raise ValueError("Rotation matrix determinant must be close to 1.")

        trace = np.trace(rotation)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (rotation[2, 1] - rotation[1, 2]) / s
            y = (rotation[0, 2] - rotation[2, 0]) / s
            z = (rotation[1, 0] - rotation[0, 1]) / s
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s

        return cls(w, x, y, z).normalize()

    @classmethod
    def from_axis_angle(
        cls,
        axis: NDArray[np.float64] | list[float] | tuple[float, ...],
        angle: float | np.floating[Any],
    ) -> "Quaternion":
        """
        Construct a quaternion from an axis-angle pair.
        """
        axis_array = np.asarray(axis, dtype=np.float64).reshape(-1)
        if axis_array.size != 3:
            raise ValueError("Axis must be a 1D array-like of length 3.")
        if not np.all(np.isfinite(axis_array)):
            raise ValueError("Axis must contain only finite values.")

        axis_norm = np.linalg.norm(axis_array)
        if np.isclose(axis_norm, 0.0, atol=ZERO_NORM_TOL):
            raise ValueError("Axis must have a non-zero norm.")

        unit_axis = axis_array / axis_norm
        half_angle = np.float64(0.5 * angle)
        c = np.cos(half_angle)
        s = np.sin(half_angle)
        return cls(
            np.float64(c),
            np.float64(unit_axis[0] * s),
            np.float64(unit_axis[1] * s),
            np.float64(unit_axis[2] * s),
        )

    def copy(self) -> "Quaternion":
        """
        Return an independent copy of the quaternion.

        The copy preserves the Hamilton convention and active-rotation semantics
        defined by the original quaternion.
        """
        return Quaternion(self.w, self.x, self.y, self.z)

    def norm(self) -> np.float64:
        """
        Return the Euclidean norm of the quaternion.

        The quaternion uses the Hamilton convention and active rotations in a
        right-handed coordinate system.
        """
        return np.float64(np.linalg.norm(self.q))

    def squared_norm(self) -> np.float64:
        """
        Return the squared Euclidean norm of the quaternion.

        The quaternion uses the Hamilton convention and active rotations in a
        right-handed coordinate system.
        """
        return np.float64(np.dot(self.q, self.q))

    def normalize(self) -> "Quaternion":
        """
        Normalize the quaternion in place.

        The quaternion follows the Hamilton convention and represents active
        rotations in a right-handed coordinate system. A zero quaternion is not
        accepted for normalization.
        """
        magnitude = self.norm()
        if np.isclose(magnitude, 0.0, atol=ZERO_NORM_TOL):
            raise ValueError("Cannot normalize a zero quaternion.")
        self.q = self.q / magnitude
        return self

    def normalized(self) -> "Quaternion":
        """
        Return a normalized copy of the quaternion.

        The returned quaternion preserves the Hamilton convention and active
        rotation semantics in a right-handed coordinate system.
        """
        return self.copy().normalize()

    def conjugate(self) -> "Quaternion":
        """
        Return the quaternion conjugate.

        The conjugate is defined for quaternions following the Hamilton
        convention and active rotations in a right-handed coordinate system.
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quaternion":
        """
        Return the multiplicative inverse of the quaternion.

        The inverse is defined for quaternions following the Hamilton
        convention and active rotations in a right-handed coordinate system.
        """
        norm_sq = self.squared_norm()
        if np.isclose(norm_sq, 0.0, atol=ZERO_NORM_TOL):
            raise ValueError("Cannot invert a zero quaternion.")
        scalar = np.float64(1.0) / np.float64(norm_sq)
        return Quaternion(
            np.float64(self.w * scalar),
            np.float64(-self.x * scalar),
            np.float64(-self.y * scalar),
            np.float64(-self.z * scalar),
        )

    def as_numpy(self) -> NDArray[np.float64]:
        """
        Return a copy of the quaternion components as a NumPy array.

        The returned array preserves the Hamilton convention storage order
        [w, x, y, z] for active rotations in a right-handed coordinate system.
        """
        return self.q.copy()

    def dot(self, other: object) -> np.float64:
        """
        Return the quaternion dot product.

        The dot product is defined for quaternions following the Hamilton
        convention and active rotations in a right-handed coordinate system.
        """
        if not isinstance(other, Quaternion):
            raise TypeError(
                "Quaternion dot product is only supported for Quaternion operands."
            )
        return np.float64(np.dot(self.q, other.q))

    def is_normalized(self, atol: float = EPS) -> bool:
        """
        Return True when the quaternion norm is within tolerance.

        The quaternion uses the Hamilton convention and active rotations in a
        right-handed coordinate system.
        """
        return bool(np.isclose(self.norm(), 1.0, atol=atol, rtol=0.0))

    def to_euler(self) -> tuple[float, float, float]:
        """
        Convert the quaternion to roll-pitch-yaw angles.

        Euler angles use the aerospace ZYX convention. The quaternion follows the
        Hamilton convention and represents active rotations in a right-handed
        coordinate system.
        """
        rotation = self.to_rotation_matrix()
        r20 = rotation[2, 0]
        r00 = rotation[0, 0]
        r10 = rotation[1, 0]
        r21 = rotation[2, 1]
        r22 = rotation[2, 2]

        pitch = np.arctan2(-r20, np.sqrt(np.maximum(r00 * r00 + r10 * r10, 0.0)))
        if np.isclose(abs(r20), 1.0, atol=ANGLE_TOL):
            roll = 0.0
            yaw = (
                np.arctan2(-rotation[0, 1], rotation[1, 1])
                if r20 < 0.0
                else np.arctan2(rotation[0, 1], -rotation[1, 1])
            )
        else:
            roll = np.arctan2(r21, r22)
            yaw = np.arctan2(r10, r00)

        return float(np.float64(roll)), float(np.float64(pitch)), float(np.float64(yaw))

    def to_rotation_matrix(self) -> NDArray[np.float64]:
        """
        Convert the quaternion to a rotation matrix.

        The quaternion follows the Hamilton convention and represents active
        rotations in a right-handed coordinate system.
        """
        q = self.normalized()
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    def to_axis_angle(self) -> tuple[NDArray[np.float64], float]:
        """
        Convert the quaternion to an axis-angle representation.

        The quaternion follows the Hamilton convention and represents active
        rotations in a right-handed coordinate system.
        """
        q = self.normalized()
        if np.isclose(q.w, 1.0, atol=ANGLE_TOL):
            return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0

        angle = 2.0 * np.arccos(np.clip(q.w, -1.0, 1.0))
        sin_half = np.sqrt(max(1.0 - q.w * q.w, 0.0))
        if np.isclose(sin_half, 0.0, atol=ZERO_NORM_TOL):
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = np.array([q.x, q.y, q.z], dtype=np.float64) / sin_half
        return axis.astype(np.float64, copy=False), float(np.float64(angle))

    def rotate_vector(
        self,
        vector: NDArray[np.float64] | list[float] | tuple[float, ...],
    ) -> NDArray[np.float64]:
        """
        Rotate a 3D vector by the quaternion.

        The quaternion follows the Hamilton convention and represents an active
        rotation in a right-handed coordinate system.
        """
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != 3:
            raise ValueError("Vector must have length 3.")

        q = self.normalized()
        vector_quaternion = Quaternion(0.0, values[0], values[1], values[2])
        rotated = q * vector_quaternion * q.inverse()
        return np.asarray(rotated.q[1:], dtype=np.float64)

    def integrate(
        self,
        angular_velocity: NDArray[np.float64] | list[float] | tuple[float, ...],
        dt: float | np.floating[Any],
    ) -> "Quaternion":
        """
        Advance the quaternion with a first-order integration step.

        The quaternion follows the Hamilton convention and represents active
        rotations in a right-handed coordinate system. The integration uses the
        first-order quaternion differential equation ``q_dot = 0.5 * q * omega``
        and renormalizes after each step.
        """
        omega = np.asarray(angular_velocity, dtype=np.float64).reshape(-1)
        if omega.size != 3:
            raise ValueError("Angular velocity must have length 3.")

        omega_quaternion = Quaternion(0.0, omega[0], omega[1], omega[2])
        q_current = self.q.copy()
        q_dot = (
            0.5
            * Quaternion(q_current[0], q_current[1], q_current[2], q_current[3])
            * omega_quaternion
        )
        self.q = q_current + q_dot.q * np.float64(dt)
        self.normalize()
        return self

    def __repr__(self) -> str:
        """
        Return a compact string representation of the quaternion.
        """
        return f"Quaternion(w={self.w:.6f}, x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})"

    def __str__(self) -> str:
        """
        Return a user-readable string representation of the quaternion.
        """
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """
        Compare two quaternions for approximate equality.
        """
        if not isinstance(other, Quaternion):
            return NotImplemented
        return bool(np.allclose(self.q, other.q, atol=EPS, rtol=0.0))

    def __neg__(self) -> "Quaternion":
        """
        Return the negation of the quaternion.
        """
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, other: object) -> "Quaternion":
        """
        Add two quaternions component-wise.
        """
        if not isinstance(other, Quaternion):
            raise TypeError(
                "Quaternion addition is only supported for Quaternion operands."
            )
        return Quaternion(
            self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
        )

    def __sub__(self, other: object) -> "Quaternion":
        """
        Subtract two quaternions component-wise.
        """
        if not isinstance(other, Quaternion):
            raise TypeError(
                "Quaternion subtraction is only supported for Quaternion operands."
            )
        return Quaternion(
            self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z
        )

    def __mul__(self, other: object) -> "Quaternion":
        """
        Multiply two quaternions or multiply by a scalar.

        Quaternion multiplication uses the Hamilton convention for active
        rotations in a right-handed coordinate system.
        """
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.w
                - self.x * other.x
                - self.y * other.y
                - self.z * other.z,
                self.w * other.x
                + self.x * other.w
                + self.y * other.z
                - self.z * other.y,
                self.w * other.y
                - self.x * other.z
                + self.y * other.w
                + self.z * other.x,
                self.w * other.z
                + self.x * other.y
                - self.y * other.x
                + self.z * other.w,
            )
        if isinstance(other, (int, float, np.integer, np.floating)):
            scalar = np.float64(other)
            return Quaternion(
                self.w * scalar, self.x * scalar, self.y * scalar, self.z * scalar
            )
        raise TypeError(
            "Quaternion multiplication is only supported for Quaternion and scalar operands."
        )

    def __rmul__(self, other: object) -> "Quaternion":
        """
        Support scalar multiplication on the left-hand side.
        """
        if isinstance(other, (int, float, np.integer, np.floating)):
            return self * other
        raise TypeError(
            "Quaternion multiplication is only supported for Quaternion and scalar operands."
        )

    def __truediv__(self, scalar: object) -> "Quaternion":
        """
        Divide the quaternion by a scalar.

        The quaternion follows the Hamilton convention and active-rotation
        semantics in a right-handed coordinate system.
        """
        if isinstance(scalar, (int, float, np.integer, np.floating)):
            scalar_value = np.float64(scalar)
            if np.isclose(scalar_value, 0.0, atol=ZERO_NORM_TOL):
                raise ZeroDivisionError("Cannot divide quaternion by a zero scalar.")
            return Quaternion(
                self.w / scalar_value,
                self.x / scalar_value,
                self.y / scalar_value,
                self.z / scalar_value,
            )
        raise TypeError("Quaternion division is only supported for scalar operands.")


__all__ = ["Quaternion"]
