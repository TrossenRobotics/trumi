"""OpenCV utilities"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import cv2
import numpy as np
import scipy.interpolate as si

# GoPro 2.7K resolution as (height, width)
GOPRO_2_7K_RESOLUTION = (2028, 2704)

# =================== intrinsics ===================


def parse_fisheye_intrinsics(json_data: dict) -> Dict[str, np.ndarray]:
    """
    Parse fisheye camera intrinsics from OpenCameraImuCalibration JSON format.

    :param json_data: Dict loaded from the calibration JSON file.
        Must have intrinsic_type == "FISHEYE".
    :return: Dict with keys DIM (np.int64 [w, h]), K (3*3 float64 camera
        matrix), and D (4*1 float64 Kannala-Brandt distortion coefficients).
    """
    if json_data["intrinsic_type"] != "FISHEYE":
        raise ValueError(
            f"Expected intrinsic_type 'FISHEYE', got '{json_data['intrinsic_type']}'"
        )
    intr_data = json_data["intrinsics"]

    # img size
    h = json_data["image_height"]
    w = json_data["image_width"]

    # pinhole parameters
    f = intr_data["focal_length"]
    px = intr_data["principal_pt_x"]
    py = intr_data["principal_pt_y"]

    # Kannala-Brandt non-linear parameters for distortion
    kb8 = [
        intr_data["radial_distortion_1"],
        intr_data["radial_distortion_2"],
        intr_data["radial_distortion_3"],
        intr_data["radial_distortion_4"],
    ]

    opencv_intr_dict = {
        "DIM": np.array([w, h], dtype=np.int64),
        "K": np.array([[f, 0, px], [0, f, py], [0, 0, 1]], dtype=np.float64),
        "D": np.array([kb8]).T,
    }
    return opencv_intr_dict


def convert_fisheye_intrinsics_resolution(
    opencv_intr_dict: Dict[str, np.ndarray], target_resolution: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """Scale fisheye intrinsics to a different resolution.

    Assuming that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in horizontal dimension.

    :param opencv_intr_dict: Intrinsics dict.
    :param target_resolution: Target (width, height) in pixels.
    :return: A new intrinsics dict with updated DIM and K for the
        target resolution.
    """
    iw, ih = opencv_intr_dict["DIM"]
    iK = opencv_intr_dict["K"]
    ifx = iK[0, 0]
    ify = iK[1, 1]
    ipx = iK[0, 2]
    ipy = iK[1, 2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([[ofx, 0, opx], [0, ofy, opy], [0, 0, 1]], dtype=np.float64)

    out_intr_dict = copy.deepcopy(opencv_intr_dict)
    out_intr_dict["DIM"] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict["K"] = oK
    return out_intr_dict


class FisheyeRectConverter:
    """Rectify fisheye images to a pinhole perspective with a given FOV."""

    def __init__(self, K, D, out_size, out_fov):
        """Pre-compute undistortion maps for the given camera and output spec.

        :param K: 3*3 fisheye camera matrix.
        :param D: 4*1 Kannala-Brandt distortion coefficients.
        :param out_size: Output image size as (width, height).
        :param out_fov: Vertical field of view of the output image in degrees.
        """
        out_size = np.array(out_size)
        # vertical fov
        out_f = (out_size[1] / 2) / np.tan(out_fov / 180 * np.pi / 2)
        out_K = np.array(
            [[out_f, 0, out_size[0] / 2], [0, out_f, out_size[1] / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), out_K, out_size, cv2.CV_16SC2
        )

        self.map1 = map1
        self.map2 = map2

    def forward(self, img):
        """Rectify a fisheye image.

        :param img: Input BGR or RGB image as a numpy array.
        :return: Undistorted image of shape (out_size[1], out_size[0], C).
        """
        rect_img = cv2.remap(
            img,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return rect_img


# ================= ArUcO tag =====================
def parse_aruco_config(aruco_config_dict: dict):
    """Parse an ArUco configuration dict into a resolved aruco_dict and marker_size_map.

    Expected YAML structure:

        aruco_dict:
            predefined: DICT_4X4_50
        marker_size_map:  # all units in meters
            default: 0.15
            12: 0.2

    :param aruco_config_dict: Dict loaded from the ArUco YAML config.
    :return: Dict with keys aruco_dict (cv2.aruco.Dictionary) and
        marker_size_map ({marker_id: size_m} for every marker in the dict).
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict["aruco_dict"])

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict["marker_size_map"]
    default_size = marker_size_map.get("default", None)

    out_marker_size_map = dict()
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map:
            size = marker_size_map[marker_id]
        if size is None:
            raise ValueError(
                f"No size defined for marker_id {marker_id} and no default provided"
            )
        out_marker_size_map[marker_id] = size

    result = {"aruco_dict": aruco_dict, "marker_size_map": out_marker_size_map}
    return result


def get_aruco_dict(predefined: str) -> cv2.aruco.Dictionary:
    """Return a predefined ArUco dictionary by name.

    :param predefined: Attribute name on cv2.aruco, e.g. "DICT_4X4_50".
    :return: The corresponding cv2.aruco.Dictionary.
    """
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, predefined))


def detect_localize_aruco_tags(
    img: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    marker_size_map: Dict[int, float],
    fisheye_intr_dict: Dict[str, np.ndarray],
    refine_subpix: bool = True,
):
    """Detect ArUco markers in a fisheye image and estimate their pose.

    :param img: Input image.
    :param aruco_dict: ArUco dictionary to use for detection.
    :param marker_size_map: {marker_id: physical_size_m} mapping.
    :param fisheye_intr_dict: Intrinsics dict.
    :param refine_subpix: If True, refine corner locations to sub-pixel
        accuracy before pose estimation.
    :return: Dict of {marker_id: {"rvec": ..., "tvec": ..., "corners": ...}}
        for each successfully localised marker.
    """
    K = fisheye_intr_dict["K"]
    D = fisheye_intr_dict["D"]
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, param)
    corners, ids, rejectedImgPoints = detector.detectMarkers(img)
    if len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue

        marker_size_m = marker_size_map[this_id]
        undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
        half = marker_size_m / 2.0
        obj_pts = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )
        img_pts = undistorted.reshape(4, 1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solveOnP(
            obj_pts, img_pts, K, np.zeros((4, 1)), flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        tag_dict[this_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": this_corners.squeeze(),
        }
    return tag_dict


def get_charuco_board(
    aruco_dict=None,
    tag_id_offset=50,
    grid_size=(8, 5),
    square_length_mm=50,
    tag_length_mm=30,
):
    """Create a ChArUco calibration board.

    :param aruco_dict: Base ArUco dictionary; markers are taken starting at
        tag_id_offset. Defaults to DICT_4X4_100.
    :param tag_id_offset: First marker ID to use from the dictionary.
    :param grid_size: Board grid as (cols, rows).
    :param square_length_mm: Checkerboard square side length in millimetres.
    :param tag_length_mm: ArUco marker side length in millimetres.
    :return: cv2.aruco.CharucoBoard instance.
    """
    if aruco_dict is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    aruco_dict = cv2.aruco.Dictionary(
        aruco_dict.bytesList[tag_id_offset:], aruco_dict.markerSize
    )
    board = cv2.aruco.CharucoBoard(
        size=grid_size,
        squareLength=square_length_mm / 1000,
        markerLength=tag_length_mm / 1000,
        dictionary=aruco_dict,
    )
    return board


def draw_charuco_board(board, dpi=300, padding_mm=15):
    """Render a ChArUco board to a grayscale image.

    :param board: cv2.aruco.CharucoBoard to render.
    :param dpi: Output resolution in dots per inch.
    :param padding_mm: White border around the board in millimetres.
    :return: Grayscale numpy array of the rendered board.
    """
    grid_size = np.array(board.getChessboardSize())
    square_length_mm = board.getSquareLength() * 1000

    mm_per_inch = 25.4
    board_size_pixel = (
        (grid_size * square_length_mm + padding_mm * 2) / mm_per_inch * dpi
    )
    board_size_pixel = board_size_pixel.round().astype(np.int64)
    padding_pixel = int(padding_mm / mm_per_inch * dpi)
    board_img = board.generateImage(outSize=board_size_pixel, marginSize=padding_pixel)
    return board_img


def get_gripper_width(tag_dict, left_id, right_id, nominal_z=0.072, z_tolerance=0.008):
    """Estimate gripper opening width from the two fingertip ArUco tags.

    :param tag_dict: Per-frame tag dict as returned by detect_localize_aruco_tags.
    :param left_id: Marker ID of the left fingertip tag.
    :param right_id: Marker ID of the right fingertip tag.
    :param nominal_z: Expected tag depth from the camera in metres.
    :param z_tolerance: Acceptable deviation from nominal_z in metres.
    :return: Estimated gripper width in metres, or None.
    """
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]["tvec"]
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]["tvec"]
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width


# =========== image mask ====================
def canonical_to_pixel_coords(coords, img_shape=GOPRO_2_7K_RESOLUTION):
    """Convert canonical (normalised) coordinates to pixel coordinates.

    :param coords: Array of shape (..., 2) in canonical (x, y) space.
    :param img_shape: Image shape as (height, width).
    :return: Array of the same shape in pixel (x, y) coordinates.
    """
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts


def pixel_coords_to_canonical(pts, img_shape=GOPRO_2_7K_RESOLUTION):
    """Convert pixel coordinates to canonical (normalised) coordinates.

    :param pts: Array of shape (..., 2) in pixel (x, y) space.
    :param img_shape: Image shape as (height, width).
    :return: Array of the same shape in canonical coordinates.
    """
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords


def get_mirror_canonical_polygon():
    """Return the left and right side-mirror polygons in canonical coordinates.

    :return: Array of shape (2, N, 2) where [0] is the left mirror
        polygon and [1] is the right mirror polygon.
    """
    # pixel coords at 2704x2028 (x, y)
    left_pts = [
        [137, 1180],
        [457, 1108],
        [502, 1180],
        [582, 1495],
        [582, 1571],
        [497, 1740],
        [471, 1824],
        [140, 1833],
    ]
    right_pts = [
        [2555, 1182],
        [2262, 1093],
        [2206, 1157],
        [2128, 1464],
        [2126, 1560],
        [2208, 1728],
        [2237, 1806],
        [2573, 1800],
    ]
    left_coords = pixel_coords_to_canonical(left_pts)
    right_coords = pixel_coords_to_canonical(right_pts)
    coords = np.stack([left_coords, right_coords])
    return coords


def get_gripper_canonical_polygon():
    """Return the gripper mask polygon in canonical coordinates.

    :return: Array of shape (1, N, 2).
    """
    # pixel coords at 2704x2028 (x, y)
    pts = [
        [0, 1834],
        [460, 1810],
        [606, 1592],
        [946, 1754],
        [1356, 1802],
        [1752, 1750],
        [2104, 1598],
        [2242, 1800],
        [2703, 1810],
        [2703, 2027],
        [0, 2027],
    ]
    coords = pixel_coords_to_canonical(pts)
    return coords[None,]  # shape (1, N, 2)


def get_finger_canonical_polygon():
    """Return the finger polygon in canonical coordinates.

    :return: Array of shape (1, N, 2).
    """
    # pixel coords at 2704x2028 (x, y)
    pts = [
        [0, 1886],
        [404, 1766],
        [577, 1537],
        [644, 1582],
        [848, 1411],
        [900, 1455],
        [1171, 1333],
        [1544, 1317],
        [1808, 1453],
        [1857, 1404],
        [2073, 1588],
        [2128, 1553],
        [2297, 1766],
        [2703, 1926],
        [2703, 2027],
        [0, 2027],
    ]
    coords = pixel_coords_to_canonical(pts)
    return coords[None,]  # shape (1, N, 2)


def draw_predefined_mask(
    img, color=(0, 0, 0), mirror=True, gripper=True, finger=True, use_aa=False
):
    """Fill predefined mask regions on an image.

    :param img: Image array of shape (H, W) or (H, W, C) to draw on.
    :param color: Fill colour, scalar for single-channel or BGR tuple for 3-channel.
    :param mirror: Mask the left and right side-mirror regions.
    :param gripper: Mask the wrist/arm gripper region.
    :param finger: Mask the fingertip region.
    :param use_aa: Use anti-aliased polygon edges (cv2.LINE_AA).
    :return: The modified image (same object as img).
    """
    all_coords = list()
    if mirror:
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        all_coords.extend(get_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_finger_canonical_polygon())

    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img, [pts], color=color, lineType=flag)
    return img


def inpaint_tag(img, corners, tag_scale=1.4, n_samples=16):
    """Inpaint an ArUco tag by filling it with the median boundary colour.

    :param img: BGR image to modify in-place.
    :param corners: (4, 2) array of tag corner pixel coordinates.
    :param tag_scale: Scale factor applied to the tag corners relative to
        their geometric centre before sampling and filling.
    :param n_samples: Number of boundary pixels to sample for the median.
    :return: The modified image.
    """
    # scale corners with respect to geometric center
    center = np.mean(corners, axis=0)
    scaled_corners = tag_scale * (corners - center) + center

    # sample pixels on the boundary to obtain median color
    sample_points = si.interp1d(
        [0, 1, 2, 3, 4], list(scaled_corners) + [scaled_corners[0]], axis=0
    )(np.linspace(0, 4, n_samples)).astype(np.int32)
    sample_colors = img[
        np.clip(sample_points[:, 1], 0, img.shape[0] - 1),
        np.clip(sample_points[:, 0], 0, img.shape[1] - 1),
    ]
    median_color = np.median(sample_colors, axis=0).astype(img.dtype)

    # draw tag with median color
    img = cv2.fillPoly(
        img, scaled_corners[None, ...].astype(np.int32), color=median_color.tolist()
    )
    return img


# =========== other utils ====================
def get_image_transform(
    in_res, out_res, crop_ratio: float = 1.0, bgr_to_rgb: bool = False
):
    """Build a centre-crop + resize transform for fixed-resolution images.

    :param in_res: Input resolution as (width, height).
    :param out_res: Output resolution as (width, height).
    :param crop_ratio: Fraction of the input height to crop (default 1.0,
        i.e. no crop).
    :param bgr_to_rgb: If True, reverse the channel axis (BGR to RGB).
    :return: A callable transform(img) that applies the crop,
        optional channel swap, and resize.
    """
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        if img.shape != (ih, iw, 3):
            raise ValueError(f"Expected image shape {(ih, iw, 3)}, got {img.shape}")
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img

    return transform
