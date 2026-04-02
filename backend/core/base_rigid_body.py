import numpy as np
import quaternion

from .math.math_types import (
    Matrix3x3,
    Quaternion,
    Scalar,
    Vector3,
    assert_inertia_tensor,
    assert_quaternion,
    assert_strictly_positive_finite_scalar,
    assert_vector3,
)


class BaseRigidBody:
    """
    Minimal rigid body state for 3D simulation.

    Conventions
    -----------
    - World frame: position, velocity, forces
    - Body frame: inertia tensor
    - Orientation: quaternion (body → world)
    - Forces/torques are accumulated per step and cleared after integration
    - Units are assumed to be standard from the international system (kg, N, etc)
    """

    # ------------------------------
    # Instance attribute annotations
    # ------------------------------

    _mass: Scalar
    _inertia_tensor: Matrix3x3

    _position: Vector3
    _velocity: Vector3
    _acceleration: Vector3

    _orientation: Quaternion
    _angular_velocity: Vector3
    _angular_acceleration: Vector3

    _accumulated_force: Vector3
    _accumulated_torque: Vector3

    # -----------
    # Constructor
    # -----------

    def __init__(
        self,
        mass: Scalar | None = None,
        inertia_tensor: Matrix3x3 | None = None,
        position: Vector3 | None = None,
        velocity: Vector3 | None = None,
        orientation: Quaternion | None = None,
        angular_velocity: Vector3 | None = None,
    ):

        # Parameters value handling

        mass = mass if mass is not None else 1.0
        inertia_tensor = inertia_tensor if inertia_tensor is not None else np.diag([0.1, 0.1, 0.1])
        position = position if position is not None else np.zeros(3)
        velocity = velocity if velocity is not None else np.zeros(3)
        orientation = orientation if orientation is not None else quaternion.quaternion(1, 0, 0, 0)
        angular_velocity = angular_velocity if angular_velocity is not None else np.zeros(3)

        acceleration = np.zeros(3)
        angular_acceleration = np.zeros(3)

        accumulated_force = np.zeros(3)
        accumulated_torque = np.zeros(3)

        # Instance variables setting

        self.mass = mass
        self.inertia_tensor = inertia_tensor
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.acceleration = acceleration
        self.angular_acceleration = angular_acceleration
        self.accumulated_force = accumulated_force
        self.accumulated_torque = accumulated_torque

    # ----------------------------------
    # Managed attributes getters/setters
    # ----------------------------------

    @property
    def mass(self) -> Scalar:
        return self._mass

    @mass.setter
    def mass(self, mass: Scalar) -> None:
        assert_strictly_positive_finite_scalar(mass)
        self._mass = mass

    @property
    def inertia_tensor(self) -> Matrix3x3:
        return self._inertia_tensor

    @inertia_tensor.setter
    def inertia_tensor(self, inertia_tensor: Matrix3x3) -> None:
        assert_inertia_tensor(inertia_tensor)
        self._inertia_tensor = inertia_tensor

    @property
    def position(self) -> Vector3:
        return self._position

    @position.setter
    def position(self, position: Vector3) -> None:
        assert_vector3(position)
        self._position = position

    @property
    def velocity(self) -> Vector3:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: Vector3) -> None:
        assert_vector3(velocity)
        self._velocity = velocity

    @property
    def acceleration(self) -> Vector3:
        return self._acceleration

    @acceleration.setter
    def acceleration(self, acceleration: Vector3) -> None:
        assert_vector3(acceleration)
        self._acceleration = acceleration

    @property
    def orientation(self) -> Quaternion:
        return self._orientation

    @orientation.setter
    def orientation(self, orientation: Quaternion) -> None:
        assert_quaternion(orientation)
        self._orientation = orientation

    @property
    def angular_velocity(self) -> Vector3:
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity: Vector3) -> None:
        assert_vector3(angular_velocity)
        self._angular_velocity = angular_velocity

    @property
    def angular_acceleration(self) -> Vector3:
        return self._angular_acceleration

    @angular_acceleration.setter
    def angular_acceleration(self, angular_acceleration: Vector3) -> None:
        assert_vector3(angular_acceleration)
        self._angular_acceleration = angular_acceleration

    @property
    def accumulated_force(self) -> Vector3:
        return self._accumulated_force

    @accumulated_force.setter
    def accumulated_force(self, accumulated_force: Vector3) -> None:
        assert_vector3(accumulated_force)
        self._accumulated_force = accumulated_force

    @property
    def accumulated_torque(self) -> Vector3:
        return self._accumulated_torque

    @accumulated_torque.setter
    def accumulated_torque(self, accumulated_torque: Vector3) -> None:
        assert_vector3(accumulated_torque)
        self._accumulated_torque = accumulated_torque

    # --------------------------
    # Linear accumulator methods
    # --------------------------

    def apply_force_world_frame(self, force_world_frame: Vector3):
        """Accumulate force in world frame."""
        assert_vector3(force_world_frame)
        self.accumulated_force += force_world_frame

    def apply_force_body_frame(self, force_body_frame: Vector3):
        """Convert body-frame force to world frame and accumulate."""
        assert_vector3(force_body_frame)
        force_world_frame = quaternion.rotate_vectors(self.orientation, force_body_frame)
        self.accumulated_force += force_world_frame

    def clear_applied_force(self):
        """Return force accumulator to zero."""
        self.accumulated_force[:] = 0
