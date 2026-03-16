import numpy as np
import quaternion

from core.base_rigid_body import BaseRigidBody


def test_accumulated_force_default_initialization():
    rb = BaseRigidBody()

    assert isinstance(
        rb._accumulated_force, np.ndarray
    ), "Accumulated force should be a numpy array after initialization"
    assert rb._accumulated_force.shape == (
        3,
    ), "Accumulated force should be of length 3 after initialization"
    np.testing.assert_array_equal(
        rb._accumulated_force, np.zeros(3), "Accumulated force should be initialized at [0,0,0]"
    )


def test_accumulated_force_clear():
    rb = BaseRigidBody()
    rb._accumulated_force = np.array([1.0, 1.0, 1.0])
    rb.clear_applied_force()

    assert isinstance(
        rb._accumulated_force, np.ndarray
    ), "Accumulated force should be a numpy array after clear"
    assert rb._accumulated_force.shape == (
        3,
    ), "Accumulated force should be of length 3 after clear"
    np.testing.assert_array_equal(
        rb._accumulated_force, np.zeros(3), "Accumulated force should be cleared to [0,0,0]"
    )


def test_apply_force_world_frame():
    rb = BaseRigidBody()

    # Tests accumulation from initialized state
    rb.apply_force_world_frame(np.ones(3))

    assert isinstance(
        rb._accumulated_force, np.ndarray
    ), "Accumulated force should be a numpy array after accumulation"
    assert rb._accumulated_force.shape == (
        3,
    ), "Accumulated force should be of length 3 after accumulation"
    np.testing.assert_array_equal(
        rb._accumulated_force, np.ones(3), "Accumulated force should accumulate to [1,1,1]"
    )

    # Tests accumulation from a none zero accumulation
    rb.apply_force_world_frame(np.array([2.0, 2.0, 2.0]))

    assert isinstance(
        rb._accumulated_force, np.ndarray
    ), "Accumulated force should be a numpy array after accumulation"
    assert rb._accumulated_force.shape == (
        3,
    ), "Accumulated force should be of length 3 after accumulation"
    np.testing.assert_array_equal(
        rb._accumulated_force,
        np.array([3.0, 3.0, 3.0]),
        "Accumulated force should accumulate to [3,3,3]",
    )


def test_apply_force_body_frame():
    rb = BaseRigidBody()

    # Tests accumulation from initialized state
    rb.apply_force_body_frame(np.ones(3))

    assert isinstance(
        rb._accumulated_force, np.ndarray
    ), "Accumulated force should be a numpy array after accumulation"
    assert rb._accumulated_force.shape == (
        3,
    ), "Accumulated force should be of length 3 after accumulation"
    np.testing.assert_array_equal(
        rb._accumulated_force, np.ones(3), "Accumulated force should accumulate to [1,1,1]"
    )

    # Tests accumulation after an orientation change
    # Note: Rotating the rigid body 90 degrees around the z axis
    angle_rad = np.deg2rad(90)
    rb._orientation = quaternion.quaternion(np.cos(angle_rad / 2), 0, 0, np.sin(angle_rad / 2))
    rb.apply_force_body_frame(np.array([1.0, 0.0, 0.0]))

    assert isinstance(
        rb._accumulated_force, np.ndarray
    ), "Accumulated force should be a numpy array after accumulation"
    assert rb._accumulated_force.shape == (
        3,
    ), "Accumulated force should be of length 3 after accumulation"
    np.testing.assert_array_equal(
        rb._accumulated_force,
        np.array([1.0, 2.0, 1.0]),
        "Accumulated force should accumulate to [1,2,1]",
    )
