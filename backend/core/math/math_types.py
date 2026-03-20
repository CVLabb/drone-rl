from typing import TypeAlias
import numpy as np
import numpy.typing as npt
import quaternion

# Primitives

Scalar: TypeAlias = float
Vector3: TypeAlias = npt.NDArray[np.float64]
Matrix3x3: TypeAlias = npt.NDArray[np.float64]
Quaternion: TypeAlias = quaternion.quaternion

# Primitive validators

def assert_scalar(x: float) -> None:
    assert isinstance(x, (float, np.floating)), f"Expected a float, got {type(x)}"
    assert np.isfinite(x), "Expected a finite number, got NaN or infinity"

def assert_vector3(v: np.ndarray):
    assert isinstance(v, np.ndarray), f"expected np.ndarray, got {type(v)}"
    assert v.shape == (3,), f"Expected (3,), got {v.shape}"
    assert v.dtype == np.float64, f"Expected dtype float64, got {v.dtype}"
    assert np.isfinite(v).all(), "Expected only finite numbers, got some NaN or infinity"

def assert_matrix3x3(m: np.ndarray):
    assert isinstance(m, np.array), f"expected np.ndarray, got {type(m)}"
    assert m.shape == (3,3), f"Expected (3,3), got {m.shape}"
    assert m.dtype == np.float64, f"Expected dtype float64, got {m.dtype}"

def assert_quaternion(q: quaternion.quaternion):
    assert isinstance(q, quaternion.quaternion), f"Expected quaternion.quaternion, got {type(q)}"
    assert abs(np.abs(q.norm()) - 1.0) < 1e-12, f"Quaternion not normalized: norm={q.norm()}"

# Special validators

def assert_inertia_tensor(it: np.ndarray):
    assert_matrix3x3(it)
    assert np.allclose(it, it.T, atol=1e-12), f"Inertia tensor not symmetric: {it}"
    assert np.all(np.linalg.eigvals(it) > 0), f"Inertia tensor not positive definite, eigenvalues={np.linalg.eigvals(it)}"

def assert_mass(m):
    assert_scalar(m)
    assert m > 0, f"Mass must be positive, got {m}"