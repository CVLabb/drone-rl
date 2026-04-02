import numpy as np
import pytest

from core.base_rigid_body import BaseRigidBody
from core.gravity_effect import GravityEffect


@pytest.mark.parametrize(
    "value,error",
    [
        ("no_param", False),  # implicit default case
        (None, False),  # explicit default case
        ([], False),  # empty list case
        ([BaseRigidBody()], False),  # single rigid body case
        ([BaseRigidBody(), BaseRigidBody()], False),  # multiple rigid bodies case
        (BaseRigidBody(), True),  # must be a list invalid case
    ],
)
def test_gravity_effect_initialization(value, error):
    if error:
        with pytest.raises(AssertionError):
            ge = GravityEffect(value)
    else:
        expected: list
        if value == "no_param":
            ge = GravityEffect()
            expected = []
        else:
            ge = GravityEffect(value)
            expected = value if value is not None else []

        assert len(ge._rigid_bodies) == len(expected)
        for rb_in, rb_out in zip(expected, ge._rigid_bodies, strict=True):
            assert rb_in is rb_out


def test_rigid_bodies_property_is_read_only():
    ge = GravityEffect([BaseRigidBody()])

    assert isinstance(ge.rigid_bodies, list)
    assert len(ge.rigid_bodies) == 1

    with pytest.raises(AttributeError):
        ge.rigid_bodies = []  # type: ignore


@pytest.mark.parametrize(
    "rigid_bodies, expected_forces",
    [
        ([], []),  # no rigid bodies case
        ([BaseRigidBody()], [np.array([0.0, 0.0, -9.81])]),  # single rigid body case
        (
            [BaseRigidBody(), BaseRigidBody(mass=2.0)],
            [np.array([0.0, 0.0, -9.81]), np.array([0.0, 0.0, -9.81 * 2.0])],
        ),  # multiple rigid body case
    ],
)
def test_gravity_effect_apply(rigid_bodies, expected_forces):
    ge = GravityEffect(rigid_bodies)
    ge.apply()

    for rb, expected_force in zip(ge._rigid_bodies, expected_forces, strict=True):
        np.testing.assert_allclose(rb.accumulated_force, expected_force)
