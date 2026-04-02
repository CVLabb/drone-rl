from typing import List

from .base_rigid_body import BaseRigidBody
from .force_effect import ForceEffect
from .math.math_types import Scalar, assert_strictly_positive_finite_scalar


class RigidBodySimEngine:
    """
    Rigid body simulation engine that applies force effects and integrates motion over time.
    """

    # ------------------------------
    # Instance attribute annotations
    # ------------------------------

    _rigid_bodies: List[BaseRigidBody]
    _force_effects: List[ForceEffect]

    # -----------
    # Constructor
    # -----------

    def __init__(
        self,
        rigid_bodies: List[BaseRigidBody] | None = None,
        force_effects: List[ForceEffect] | None = None,
    ):
        # Default value handling
        rigid_bodies = rigid_bodies if rigid_bodies is not None else []
        force_effects = force_effects if force_effects is not None else []

        # Assert parameters' validity
        assert isinstance(rigid_bodies, list)
        for rb in rigid_bodies:
            assert isinstance(rb, BaseRigidBody)

        assert isinstance(force_effects, list)
        for fe in force_effects:
            assert isinstance(fe, ForceEffect)

        # Attributes allocation
        self._rigid_bodies = []
        self._rigid_bodies.extend(rigid_bodies)

        self._force_effects = []
        self._force_effects.extend(force_effects)

    # --------------------
    # Read only attributes
    # --------------------

    @property
    def rigid_bodies(self) -> List[BaseRigidBody]:
        return self._rigid_bodies

    @property
    def force_effects(self) -> List[ForceEffect]:
        return self._force_effects

    # --------------
    # Public methods
    # --------------

    def step(self, delta_t: Scalar) -> None:
        """Advance the simulation by a single timestep."""

        assert_strictly_positive_finite_scalar(delta_t)

        for rb in self._rigid_bodies:
            rb.clear_applied_force()

        for fe in self._force_effects:
            fe.apply()

        self._rigid_bodies_integration(delta_t)

    # ---------------
    # Private methods
    # ---------------

    def _rigid_bodies_integration(self, delta_t: Scalar) -> None:
        """Integrate all rigid bodies over the given timestep."""
        for rb in self._rigid_bodies:
            self._rigid_body_linear_integration(rb, delta_t)
            self._rigid_body_angular_integration(rb, delta_t)

    def _rigid_body_linear_integration(self, rb: BaseRigidBody, delta_t: Scalar) -> None:
        """
        Perform linear integration (velocity and position update) for a rigid body
        using semi-implicit Euler method.
        """
        rb.acceleration = rb.accumulated_force / rb.mass
        rb.velocity += rb.acceleration * delta_t
        rb.position += rb.velocity * delta_t

    def _rigid_body_angular_integration(self, rb: BaseRigidBody, delta_t: Scalar) -> None:
        pass
