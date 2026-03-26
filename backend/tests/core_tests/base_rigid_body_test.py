import sys

import numpy as np
import pytest
import quaternion

from core.base_rigid_body import BaseRigidBody
from core.math.math_types import (
    assert_quaternions_equal,
)

# -----------------
# Constructor tests
# -----------------


@pytest.mark.parametrize(
    "value,expected,error",
    [
        ("no_param", 1.0, False),  # implicit default
        (None, 1.0, False),  # explicit default
        (2.0, 2.0, False),  # normal range mass value
        (sys.float_info.max, sys.float_info.max, False),  # valid extremely big mass value
        (sys.float_info.min, sys.float_info.min, False),  # valid extremely small mass value
        (0.0, None, True),  # invalid null mass
        (-1.0, None, True),  # invalid negative mass
        (1, None, True),  # invalid type mass
        (float("nan"), None, True),  # invalid nan mass
        (np.inf, None, True),  # invalid infinity mass
    ],
)
def test_mass_initialization(value, expected, error):
    if error:
        with pytest.raises(AssertionError):
            BaseRigidBody(mass=value)
    else:
        if value == "no_param":
            rb = BaseRigidBody()
        else:
            rb = BaseRigidBody(mass=value)
        assert rb.mass == expected


@pytest.mark.parametrize(
    "value,expected,error",
    [
        ("no_param", np.diag([0.1, 0.1, 0.1]), False),  # implicit default
        (None, np.diag([0.1, 0.1, 0.1]), False),  # explicit default
        (np.diag([0.2, 0.2, 0.2]), np.diag([0.2, 0.2, 0.2]), False),  # regular valid tensor
        (np.zeros((3, 3)), np.zeros((3, 3)), False),  # technically valid zero mass tensor
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], None, True),  # not a np array
        (np.array([1, 1, 1]), None, True),  # invalid tensor shape lower dimension
        (
            np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
            None,
            True,
        ),  # invalid tensor shape higher rank
        (np.diag(np.array([1, 1, 1], dtype=int)), None, True),  # invalid dtype tensor
        (np.diag([float("nan"), 1, 1]), None, True),  # invalid nan containing tensor
        (np.diag([np.inf, 1, 1]), None, True),  # invalid infinity containing tensor
        (np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]]), None, True),  # invalid asymmetric tensor
        (np.diag([1, -1, 1]), None, True),  # invalid negative eigenvalue tensor
    ],
)
def test_inertia_tensor_initialization(value, expected, error):
    if error:
        with pytest.raises(AssertionError):
            BaseRigidBody(inertia_tensor=value)
    else:
        if isinstance(value, str) and value == "no_param":
            rb = BaseRigidBody()
        else:
            rb = BaseRigidBody(inertia_tensor=value)
        np.testing.assert_array_equal(rb.inertia_tensor, expected)


@pytest.mark.parametrize("attr", ["position", "velocity", "angular_velocity"])
@pytest.mark.parametrize(
    "value,expected,error",
    [
        ("no_param", np.array([0.0, 0.0, 0.0]), False),  # implicit default
        (None, np.array([0.0, 0.0, 0.0]), False),  # explicit default
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), False),  # simple position
        (
            np.array([sys.float_info.max, sys.float_info.max, sys.float_info.max]),
            np.array([sys.float_info.max, sys.float_info.max, sys.float_info.max]),
            False,
        ),  # valid extremely large vector
        (
            np.array([sys.float_info.min, sys.float_info.min, sys.float_info.min]),
            np.array([sys.float_info.min, sys.float_info.min, sys.float_info.min]),
            False,
        ),  # valid extremely small vector
        ([1.0, 1.0, 1.0], None, True),  # not a np array
        (np.array([0.0, 0.0]), None, True),  # invalid vector shape too short
        (np.array([[1.0, 1.0, 1.0]]), None, True),  # invalid vector shape wrong rank
        (np.array([1, 1, 1], dtype=int), None, True),  # invalid dtype vector
        (np.array([float("nan"), 0, 0]), None, True),  # invalid nan containing vector
        (np.array([np.inf, 0, 0]), None, True),  # invalid infinity containing vector
    ],
)
def test_assignable_vector3_attributes_initialization(attr, value, expected, error):
    kwargs = {}
    if not isinstance(value, str) or value != "no_param":
        kwargs[attr] = value

    if error:
        with pytest.raises(AssertionError):
            BaseRigidBody(**kwargs)
    else:
        rb = BaseRigidBody(**kwargs)
        np.testing.assert_array_equal(getattr(rb, attr), expected)


@pytest.mark.parametrize(
    "attr", ["acceleration", "angular_acceleration", "accumulated_force", "accumulated_torque"]
)
def test_default_only_vector3_attributes_initialization(attr):
    rb = BaseRigidBody()
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_equal(getattr(rb, attr), expected)


@pytest.mark.parametrize(
    "value,expected,error",
    [
        ("no_param", quaternion.quaternion(1, 0, 0, 0), False),  # implicit default
        (None, quaternion.quaternion(1, 0, 0, 0), False),  # explicit default
        (
            quaternion.quaternion(0.7071067811865476, 0.7071067811865476, 0, 0),
            quaternion.quaternion(0.7071067811865476, 0.7071067811865476, 0, 0),
            False,
        ),  # usual quaternion
        (
            quaternion.quaternion(-1, 0, 0, 0),
            quaternion.quaternion(-1, 0, 0, 0),
            False,
        ),  # equivalent negative quaternion
        (
            quaternion.quaternion(1 + 0.25e-12, 0, 0, 0),
            quaternion.quaternion(1 + 0.25e-12, 0, 0, 0),
            False,
        ),  # normalization tolerance over boundary
        (
            quaternion.quaternion(1 - 0.25e-12, 0, 0, 0),
            quaternion.quaternion(1 - 0.25e-12, 0, 0, 0),
            False,
        ),  # normalization tolerance under boundary
        ((1, 0, 0, 0), None, True),  # invalid not a quaternion instance quaternion
        (quaternion.quaternion(0, 0, 0, 0), None, True),  # invalid degenerate quaternion
        (quaternion.quaternion(1 + 1e-12, 0, 0, 0), None, True),  # over-normalized quaternion
        (quaternion.quaternion(1 - 1e-12, 0, 0, 0), None, True),  # under-normalized quaternion
        (quaternion.quaternion(float("nan"), 0, 0, 0), None, True),  # invalid quaternion with nan
        (quaternion.quaternion(np.inf, 0, 0, 0), None, True),  # invalid quaternion with infinity
    ],
)
def test_orientation_initialization(value, expected, error):
    if error:
        with pytest.raises(AssertionError):
            BaseRigidBody(orientation=value)
    else:
        if isinstance(value, str) and value == "no_param":
            rb = BaseRigidBody()
        else:
            rb = BaseRigidBody(orientation=value)
        assert_quaternions_equal(rb.orientation, expected)


# ------------
# Setter tests
# ------------


@pytest.mark.parametrize(
    "value,expected,error",
    [
        (2.0, 2.0, False),  # normal range mass value
        (sys.float_info.max, sys.float_info.max, False),  # valid extremely big mass value
        (sys.float_info.min, sys.float_info.min, False),  # valid extremely small mass value
        (0.0, None, True),  # invalid null mass
        (-1.0, None, True),  # invalid negative mass
        (1, None, True),  # invalid type mass
        (float("nan"), None, True),  # invalid nan mass
        (np.inf, None, True),  # invalid infinity mass
    ],
)
def test_mass_setter(value, expected, error):
    rb = BaseRigidBody()
    if error:
        with pytest.raises(AssertionError):
            rb.mass = value
    else:
        rb.mass = value
        assert rb.mass == expected


@pytest.mark.parametrize(
    "value,expected,error",
    [
        (np.diag([0.2, 0.2, 0.2]), np.diag([0.2, 0.2, 0.2]), False),  # regular valid tensor
        (np.zeros((3, 3)), np.zeros((3, 3)), False),  # technically valid zero mass tensor
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], None, True),  # not a np array
        (np.array([1, 1, 1]), None, True),  # invalid tensor shape lower dimension
        (
            np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
            None,
            True,
        ),  # invalid tensor shape higher rank
        (np.diag(np.array([1, 1, 1], dtype=int)), None, True),  # invalid dtype tensor
        (np.diag([float("nan"), 1, 1]), None, True),  # invalid nan containing tensor
        (np.diag([np.inf, 1, 1]), None, True),  # invalid infinity containing tensor
        (np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]]), None, True),  # invalid asymmetric tensor
        (np.diag([1, -1, 1]), None, True),  # invalid negative eigenvalue tensor
    ],
)
def test_inertia_tensor_setter(value, expected, error):
    rb = BaseRigidBody()
    if error:
        with pytest.raises(AssertionError):
            rb.inertia_tensor = value
    else:
        rb.inertia_tensor = value
        np.testing.assert_array_equal(rb.inertia_tensor, expected)


@pytest.mark.parametrize(
    "attr",
    [
        "position",
        "velocity",
        "acceleration",
        "angular_velocity",
        "angular_acceleration",
        "accumulated_force",
        "accumulated_torque",
    ],
)
@pytest.mark.parametrize(
    "value,expected,error",
    [
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), False),  # simple position
        (
            np.array([sys.float_info.max, sys.float_info.max, sys.float_info.max]),
            np.array([sys.float_info.max, sys.float_info.max, sys.float_info.max]),
            False,
        ),  # valid extremely large vector
        (
            np.array([sys.float_info.min, sys.float_info.min, sys.float_info.min]),
            np.array([sys.float_info.min, sys.float_info.min, sys.float_info.min]),
            False,
        ),  # valid extremely small vector
        ([1.0, 1.0, 1.0], None, True),  # not a np array
        (np.array([0.0, 0.0]), None, True),  # invalid vector shape too short
        (np.array([[1.0, 1.0, 1.0]]), None, True),  # invalid vector shape wrong rank
        (np.array([1, 1, 1], dtype=int), None, True),  # invalid dtype vector
        (np.array([float("nan"), 0, 0]), None, True),  # invalid nan containing vector
        (np.array([np.inf, 0, 0]), None, True),  # invalid infinity containing vector
    ],
)
def test_vector3_attributes_setters(attr, value, expected, error):
    rb = BaseRigidBody()
    if error:
        with pytest.raises(AssertionError):
            setattr(rb, attr, value)
    else:
        setattr(rb, attr, value)
        np.testing.assert_array_equal(getattr(rb, attr), expected)


@pytest.mark.parametrize(
    "value,expected,error",
    [
        (
            quaternion.quaternion(0.7071067811865476, 0.7071067811865476, 0, 0),
            quaternion.quaternion(0.7071067811865476, 0.7071067811865476, 0, 0),
            False,
        ),  # usual quaternion
        (
            quaternion.quaternion(-1, 0, 0, 0),
            quaternion.quaternion(-1, 0, 0, 0),
            False,
        ),  # equivalent negative quaternion
        (
            quaternion.quaternion(1 + 0.25e-12, 0, 0, 0),
            quaternion.quaternion(1 + 0.25e-12, 0, 0, 0),
            False,
        ),  # normalization tolerance over boundary
        (
            quaternion.quaternion(1 - 0.25e-12, 0, 0, 0),
            quaternion.quaternion(1 - 0.25e-12, 0, 0, 0),
            False,
        ),  # normalization tolerance under boundary
        ((1, 0, 0, 0), None, True),  # invalid not a quaternion instance quaternion
        (quaternion.quaternion(0, 0, 0, 0), None, True),  # invalid degenerate quaternion
        (quaternion.quaternion(1 + 1e-12, 0, 0, 0), None, True),  # over-normalized quaternion
        (quaternion.quaternion(1 - 1e-12, 0, 0, 0), None, True),  # under-normalized quaternion
        (quaternion.quaternion(float("nan"), 0, 0, 0), None, True),  # invalid quaternion with nan
        (quaternion.quaternion(np.inf, 0, 0, 0), None, True),  # invalid quaternion with infinity
    ],
)
def test_orientation_setter(value, expected, error):
    rb = BaseRigidBody()
    if error:
        with pytest.raises(AssertionError):
            rb.orientation = value
    else:
        rb.orientation = value
        assert_quaternions_equal(rb.orientation, expected)


# ------------
# Linear force accumulator tests
# ------------


def test_accumulated_force_default_initialization():
    rb = BaseRigidBody()

    assert isinstance(rb._accumulated_force, np.ndarray), (
        "Accumulated force should be a numpy array after initialization"
    )
    assert rb._accumulated_force.shape == (3,), (
        "Accumulated force should be of length 3 after initialization"
    )
    np.testing.assert_array_equal(
        rb._accumulated_force, np.zeros(3), "Accumulated force should be initialized at [0,0,0]"
    )


def test_accumulated_force_clear():
    rb = BaseRigidBody()
    rb._accumulated_force = np.array([1.0, 1.0, 1.0])
    rb.clear_applied_force()

    assert isinstance(rb._accumulated_force, np.ndarray), (
        "Accumulated force should be a numpy array after clear"
    )
    assert rb._accumulated_force.shape == (3,), (
        "Accumulated force should be of length 3 after clear"
    )
    np.testing.assert_array_equal(
        rb._accumulated_force, np.zeros(3), "Accumulated force should be cleared to [0,0,0]"
    )


def test_apply_force_world_frame():
    rb = BaseRigidBody()

    # Tests accumulation from initialized state
    rb.apply_force_world_frame(np.ones(3))

    assert isinstance(rb._accumulated_force, np.ndarray), (
        "Accumulated force should be a numpy array after accumulation"
    )
    assert rb._accumulated_force.shape == (3,), (
        "Accumulated force should be of length 3 after accumulation"
    )
    np.testing.assert_array_equal(
        rb._accumulated_force, np.ones(3), "Accumulated force should accumulate to [1,1,1]"
    )

    # Tests accumulation from a none zero accumulation
    rb.apply_force_world_frame(np.array([2.0, 2.0, 2.0]))

    assert isinstance(rb._accumulated_force, np.ndarray), (
        "Accumulated force should be a numpy array after accumulation"
    )
    assert rb._accumulated_force.shape == (3,), (
        "Accumulated force should be of length 3 after accumulation"
    )
    np.testing.assert_array_equal(
        rb._accumulated_force,
        np.array([3.0, 3.0, 3.0]),
        "Accumulated force should accumulate to [3,3,3]",
    )


def test_apply_force_body_frame():
    rb = BaseRigidBody()

    # Tests accumulation from initialized state
    rb.apply_force_body_frame(np.ones(3))

    assert isinstance(rb._accumulated_force, np.ndarray), (
        "Accumulated force should be a numpy array after accumulation"
    )
    assert rb._accumulated_force.shape == (3,), (
        "Accumulated force should be of length 3 after accumulation"
    )
    np.testing.assert_array_equal(
        rb._accumulated_force, np.ones(3), "Accumulated force should accumulate to [1,1,1]"
    )

    # Tests accumulation after an orientation change
    # Note: Rotating the rigid body 90 degrees around the z axis
    angle_rad = np.deg2rad(90)
    rb._orientation = quaternion.quaternion(np.cos(angle_rad / 2), 0, 0, np.sin(angle_rad / 2))
    rb.apply_force_body_frame(np.array([1.0, 0.0, 0.0]))

    assert isinstance(rb._accumulated_force, np.ndarray), (
        "Accumulated force should be a numpy array after accumulation"
    )
    assert rb._accumulated_force.shape == (3,), (
        "Accumulated force should be of length 3 after accumulation"
    )
    np.testing.assert_array_equal(
        rb._accumulated_force,
        np.array([1.0, 2.0, 1.0]),
        "Accumulated force should accumulate to [1,2,1]",
    )
