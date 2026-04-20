"""Rigid-body pose utilities.

6-DoF convention
----------------
Pose vectors are laid out as [x, y, z, rx, ry, rz] where
(x, y, z) is the translation and (rx, ry, rz) is the
rotation in axis-angle (rotation-vector) form as used by
scipy.spatial.transform.Rotation.from_rotvec.

Transform matrices are 4x4 homogeneous matrices in the standard
right-hand convention::

    [R | t]
    [0 | 1]

where R is a 3x3 rotation matrix and t is a 3-vector
translation.  tx_A_B denotes a transform that maps points
from frame B into frame A.
"""

import numpy as np
import scipy.spatial.transform as st


def pos_rot_to_mat(pos, rot):
    """Convert position array and Rotation to a 4×4 homogeneous transform matrix.

    :param pos: Translation vector(s), shape (..., 3).
    :param rot: scipy.spatial.transform.Rotation with batch shape (...).
    :return: Homogeneous transform matrix, shape (..., 4, 4).
    """
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., 3, 3] = 1
    return mat


def mat_to_pos_rot(mat):
    """Decompose a 4×4 homogeneous matrix into (position, Rotation).

    :param mat: Homogeneous transform matrix, shape (..., 4, 4).
    :return: Tuple of (position array shape (..., 3), Rotation with batch shape (...)).
    """
    pos = mat[..., :3, 3] / mat[..., 3, 3][..., None]
    rot = st.Rotation.from_matrix(mat[..., :3, :3])
    return pos, rot


def pos_rot_to_pose(pos, rot):
    """Pack position and Rotation into a 6-DoF pose vector [x, y, z, rx, ry, rz].

    :param pos: Translation vector(s), shape (..., 3).
    :param rot: scipy.spatial.transform.Rotation with batch shape (...).
    :return: Pose vector(s), shape (..., 6).
    """
    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def pose_to_pos_rot(pose):
    """Unpack a 6-DoF pose vector into (position, Rotation).

    :param pose: Pose vector(s), shape (..., 6).
    :return: Tuple of (position array shape (..., 3), Rotation with batch shape (...)).
    """
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])
    return pos, rot


def pose_to_mat(pose):
    """Convert a 6-DoF pose vector to a 4×4 homogeneous transform matrix.

    :param pose: Pose vector(s), shape (..., 6).
    :return: Homogeneous transform matrix, shape (..., 4, 4).
    """
    return pos_rot_to_mat(*pose_to_pos_rot(pose))


def mat_to_pose(mat):
    """Convert a 4×4 homogeneous transform matrix to a 6-DoF pose vector.

    :param mat: Homogeneous transform matrix, shape (..., 4, 4).
    :return: Pose vector(s), shape (..., 6).
    """
    return pos_rot_to_pose(*mat_to_pos_rot(mat))
