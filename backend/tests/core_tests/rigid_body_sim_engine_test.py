import numpy as np
import pytest

from core.base_rigid_body import BaseRigidBody
from core.force_effect import ForceEffect
from core.gravity_effect import GravityEffect
from core.rigid_body_sim_engine import RigidBodySimEngine

# -----------------
# Constructor tests
# -----------------


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
def test_engine_initialization_with_rigid_bodies(value, error):
    if error:
        with pytest.raises(AssertionError):
            engine = RigidBodySimEngine(rigid_bodies=value)
    else:
        if value == "no_param":
            engine = RigidBodySimEngine()
            expected: list = []
        else:
            engine = RigidBodySimEngine(rigid_bodies=value)
            expected = value if value is not None else []

        assert len(engine._rigid_bodies) == len(expected)
        for rb_in, rb_out in zip(expected, engine._rigid_bodies, strict=True):
            assert rb_in is rb_out


@pytest.mark.parametrize(
    "value,error",
    [
        ("no_param", False),  # implicit default case
        (None, False),  # explicit default case
        ([], False),  # empty list case
        ([GravityEffect()], False),  # single force effect case
        ([GravityEffect(), GravityEffect()], False),  # multiple force effect case
        (GravityEffect(), True),  # must be a list invalid case
    ],
)
def test_engine_initialization_with_force_effects(value, error):
    if error:
        with pytest.raises(AssertionError):
            engine = RigidBodySimEngine(force_effects=value)
    else:
        expected: list
        if value == "no_param":
            engine = RigidBodySimEngine()
            expected = []
        else:
            engine = RigidBodySimEngine(force_effects=value)
            expected = value if value is not None else []

        assert len(engine._force_effects) == len(expected)
        for fe_in, fe_out in zip(expected, engine._force_effects, strict=True):
            assert fe_in is fe_out


# ----------
# Step tests
# ----------


@pytest.mark.parametrize("dt", [0, -1, float("inf"), float("nan")])
def test_step_invalid_delta_t(dt):
    engine = RigidBodySimEngine()
    with pytest.raises(AssertionError):
        engine.step(dt)


@pytest.mark.parametrize(
    "rigid_bodies, force_effects_number, expected_velocity, expected_position",
    [
        ([BaseRigidBody()], 1, [np.array([0, 0, -9.81])], [np.array([0, 0, -9.81])]),
        (
            [BaseRigidBody(velocity=np.array([1.0, 1.0, 1.0]))],
            1,
            [np.array([1.0, 1.0, -8.81])],
            [np.array([1.0, 1.0, -8.81])],
        ),
        (
            [BaseRigidBody(position=np.array([1.0, 1.0, 1.0]))],
            1,
            [np.array([0, 0, -9.81])],
            [np.array([1.0, 1.0, -8.81])],
        ),
        (
            [BaseRigidBody(), BaseRigidBody(mass=2.0)],
            1,
            [np.array([0, 0, -9.81]), np.array([0, 0, -9.81])],
            [np.array([0, 0, -9.81]), np.array([0, 0, -9.81])],
        ),
        ([BaseRigidBody()], 2, [np.array([0, 0, -9.81]) * 2.0], [np.array([0, 0, -9.81]) * 2.0]),
        ([], 0, [], []),
    ],
)
def test_step_linear_integration(
    rigid_bodies, force_effects_number, expected_velocity, expected_position
):
    force_effects: list[ForceEffect] = []
    for _i in range(force_effects_number):
        force_effects.append(GravityEffect(rigid_bodies=rigid_bodies))

    engine = RigidBodySimEngine(rigid_bodies=rigid_bodies, force_effects=force_effects)
    engine.step(1.0)

    for rb, ev, ep in zip(rigid_bodies, expected_velocity, expected_position, strict=True):
        np.testing.assert_allclose(rb.velocity, ev, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(rb.position, ep, rtol=1e-5, atol=1e-8)


def test_step_repeated_behavior():
    rb = BaseRigidBody()
    fe = GravityEffect(rigid_bodies=[rb])
    engine = RigidBodySimEngine(rigid_bodies=[rb], force_effects=[fe])

    position0 = rb.position
    velocity0 = rb.velocity

    engine.step(1.0)
    position1 = rb.position
    velocity1 = rb.velocity

    engine.step(1.0)
    position2 = rb.position
    velocity2 = rb.velocity

    np.testing.assert_allclose(position1 - position0, position2 - position1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(velocity1 - velocity0, velocity2 - velocity1, rtol=1e-5, atol=1e-8)
