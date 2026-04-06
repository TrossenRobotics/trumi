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


def transform_pose(tx, pose):
    """Apply transform tx_new_old to pose tx_old_obj, returning tx_new_obj.

    :param tx: 4×4 homogeneous transform matrix.
    :param pose: 6-DoF pose vector, shape (6,).
    :return: Transformed 6-DoF pose vector, shape (6,).
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose


def transform_point(tx, point):
    """Apply a 4×4 transform to a 3D point (or batch of points).

    :param tx: 4×4 homogeneous transform matrix.
    :param point: 3D point(s), shape (..., 3).
    :return: Transformed 3D point(s), shape (..., 3).
    """
    return point @ tx[:3, :3].T + tx[:3, 3]


def project_point(k, point):
    """Project 3D points to 2D pixel coordinates using intrinsic matrix k.

    :param k: 3×3 camera intrinsic matrix.
    :param point: 3D point(s) in camera frame, shape (..., 3).
    :return: 2D pixel coordinates, shape (..., 2).
    """
    x = point @ k.T
    uv = x[..., :2] / x[..., [2]]
    return uv


def apply_delta_pose(pose, delta_pose):
    """Add a delta to a 6-DoF pose: translate additively, rotate multiplicatively.

    :param pose: Base 6-DoF pose vector, shape (6,).
    :param delta_pose: Delta 6-DoF pose vector, shape (6,).
    :return: Updated 6-DoF pose vector, shape (6,).
    """
    new_pose = np.zeros_like(pose)
    new_pose[:3] = pose[:3] + delta_pose[:3]
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()
    return new_pose


def normalize(vec, eps=1e-12):
    """L2-normalize vectors along the last axis.

    :param vec: Input array, shape (..., N).
    :param eps: Small value to avoid division by zero.
    :return: L2-normalized array, shape (..., N).
    """
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return vec / norm


def rot_from_directions(from_vec, to_vec):
    """Compute the minimal Rotation that maps from_vec onto to_vec.

    :param from_vec: Source direction vector, shape (3,).
    :param to_vec: Target direction vector, shape (3,).
    :return: scipy.spatial.transform.Rotation mapping from_vec to to_vec.
    """
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    # Use sum-of-products for correctness with batched inputs.
    dot = np.clip(np.sum(from_vec * to_vec, axis=-1), -1.0, 1.0)
    axis = np.cross(from_vec, to_vec)
    # Handle antiparallel case where cross product is ~0 but angle is ~pi.
    if np.linalg.norm(axis) < 1e-6:
        # Pick a stable perpendicular axis.
        ref = (
            np.array([1.0, 0.0, 0.0])
            if abs(from_vec[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(from_vec, ref)
    axis = normalize(axis)
    angle = np.arccos(dot)
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot


def rot6d_to_mat(d6):
    """Convert a 6D continuous rotation representation to a 3×3 rotation matrix.

    :param d6: 6D rotation representation, shape (..., 6).
    :return: Rotation matrix, shape (..., 3, 3).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out


def mat_to_rot6d(mat):
    """Convert a 3×3 rotation matrix to its 6D continuous representation.

    :param mat: Rotation matrix, shape (..., 3, 3).
    :return: 6D rotation representation, shape (..., 6).
    """
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def mat_to_pose9d(mat):
    """Convert a 4×4 transform to a 9D pose [pos(3) + rot6d(6)] vector.

    :param mat: Homogeneous transform matrix, shape (..., 4, 4).
    :return: 9D pose vector, shape (..., 9).
    """
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d9 = np.concatenate([pos, d6], axis=-1)
    return d9


def pose9d_to_mat(d9):
    """Convert a 9D pose vector [pos(3) + rot6d(6)] to a 4×4 transform matrix.

    :param d9: 9D pose vector, shape (..., 9).
    :return: Homogeneous transform matrix, shape (..., 4, 4).
    """
    pos = d9[..., :3]
    d6 = d9[..., 3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d9.shape[:-1] + (4, 4), dtype=d9.dtype)
    out[..., :3, :3] = rotmat
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1
    return out
