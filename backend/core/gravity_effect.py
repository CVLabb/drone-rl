from typing import List

import numpy as np

from .base_rigid_body import BaseRigidBody
from .force_effect import ForceEffect
from .math.math_types import Scalar, Vector3


class GravityEffect(ForceEffect):
    """
    Applies a constant gravitational force to a collection of rigid bodies.

    The force follows F = m * g, where g has fixed magnitude and a constant
    world-space direction (downward along -Z).

    Forces are accumulated in world coordinates via each rigid body's interface.
    """

    # ---------------------
    # Class level constants
    # ---------------------

    G: Scalar = 9.81
    GRAVITY_DIRECTION: Vector3 = np.array([0.0, 0.0, -1.0])

    # ------------------------------
    # Instance attribute annotations
    # ------------------------------

    _rigid_bodies: List[BaseRigidBody]

    # -----------
    # Constructor
    # -----------

    def __init__(self, rigid_bodies: List[BaseRigidBody] | None = None):
        rigid_bodies = rigid_bodies if rigid_bodies is not None else []

        assert isinstance(rigid_bodies, list)
        for rb in rigid_bodies:
            assert isinstance(rb, BaseRigidBody)

        self._rigid_bodies = []
        self._rigid_bodies.extend(rigid_bodies)

    # --------------------
    # Read only attributes
    # --------------------

    @property
    def rigid_bodies(self) -> List[BaseRigidBody]:
        return self._rigid_bodies

    # -------------
    # Effect method
    # -------------

    def apply(self):
        """
        Applies gravitational force to all registered rigid bodies in world frame.
        """
        # Precompute gravitational acceleration vector
        g_vec = self.G * self.GRAVITY_DIRECTION

        for rb in self._rigid_bodies:
            # Linear accumulation
            gravity = rb.mass * g_vec
            rb.apply_force_world_frame(gravity)
