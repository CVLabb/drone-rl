import numpy as np
import numpy.typing as npt
import quaternion


class BaseRigidBody:
    _orientation: quaternion.quaternion
    _accumulated_force: npt.NDArray[np.float64]

    def __init__(self):
        self._accumulated_force = np.zeros(3)
        self._orientation = quaternion.quaternion(1, 0, 0, 0)

    def apply_force_world_frame(self, force_world_frame: npt.NDArray[np.float64]):
        assert force_world_frame.shape == (3,), "Force must be a 3D vector"
        self._accumulated_force += force_world_frame

    def apply_force_body_frame(self, force_body_frame: npt.NDArray[np.float64]):
        assert force_body_frame.shape == (3,), "Force must be a 3D vector"
        force_world_frame = quaternion.rotate_vectors(self._orientation, force_body_frame)
        self._accumulated_force += force_world_frame

    def clear_applied_force(self):
        self._accumulated_force[:] = 0
