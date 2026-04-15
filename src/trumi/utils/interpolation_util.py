import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_interp1d(t, x):
    """Create a 1-D linear interpolator with flat extrapolation at the boundaries.

    :param t: 1-D array of sample times.
    :param x: Array of values with shape (N, ...) corresponding to t.
    :return: scipy interp1d interpolator.
    """
    gripper_interp = si.interp1d(
        t, x, axis=0, bounds_error=False, fill_value=(x[0], x[-1])
    )
    return gripper_interp


class PoseInterpolator:
    """Interpolate 6-DOF poses (position + axis-angle) over time.

    Position is interpolated linearly; rotation is interpolated via SLERP.
    Queries outside the time range are clamped to the nearest endpoint.
    """

    def __init__(self, t, x):
        pos = x[:, :3]
        rot = st.Rotation.from_rotvec(x[:, 3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)

    @property
    def x(self):
        return self.pos_interp.x

    def __call__(self, t):
        """Interpolate pose at the given time.

        :param t: Scalar or array of query times.
        :return: np.ndarray of shape (..., 6) — [x, y, z, rx, ry, rz].
        """
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose


def get_gripper_calibration_interpolator(aruco_measured_width, aruco_actual_width):
    """Build a calibration interpolator from raw ArUco widths to calibrated gripper widths.

    Maps measured ArUco tag separation to actual gripper opening (meters), with the
    minimum actual width subtracted so fully-closed = 0.

    :param aruco_measured_width: Array of measured tag separations (m), e.g. [min, max].
    :param aruco_actual_width: Corresponding actual gripper widths (m), e.g. [min, max].
    :return: interp1d interpolator from measured width to calibrated opening width.
    """
    aruco_measured_width = np.array(aruco_measured_width)
    aruco_actual_width = np.array(aruco_actual_width)
    if len(aruco_measured_width) != len(aruco_actual_width):
        raise ValueError(
            f"aruco_measured_width and aruco_actual_width must have the same length, "
            f"got {len(aruco_measured_width)} and {len(aruco_actual_width)}"
        )
    if len(aruco_actual_width) < 2:
        raise ValueError(
            f"At least 2 calibration points required, got {len(aruco_actual_width)}"
        )
    aruco_min_width = np.min(aruco_actual_width)
    gripper_actual_width = aruco_actual_width - aruco_min_width
    interp = get_interp1d(aruco_measured_width, gripper_actual_width)
    return interp
