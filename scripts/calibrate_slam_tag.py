"""Calibrate the mapping ArUco tag pose in the SLAM map coordinate frame.

:Steps:
    1. Load tag_detection.pkl and camera_trajectory.csv from the mapping video.
    2. For each valid camera pose where the mapping tag is visible, compute
       tx_slam_tag = tx_slam_cam * tx_cam_tag.
    3. Filter outliers by depth, image-center distance, and 90th-percentile
       geometric-median distance.
    4. Save the most representative tx_slam_tag as a JSON file.

:Usage:
    uv run python scripts/calibrate_slam_tag.py \\
        -d path/to/tag_detection.pkl \\
        -c path/to/camera_trajectory.csv \\
        -o path/to/tx_slam_tag.json
"""

import json
import logging
import pathlib
import pickle

import click
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from skfda.exploratory.stats import geometric_median

from trumi.utils.cv_util import GOPRO_2_7K_RESOLUTION
from trumi.utils.pose_util import pose_to_mat

logger = logging.getLogger(__name__)

# Min/max tag distance from camera (metres).
# Too close: ArUco pose estimation degrades. Too far: detection is noisy.
MIN_TAG_DIST_M = 0.3
MAX_TAG_DIST_M = 2.0
# Max allowed distance from image center as a fraction of half-height.
# Beyond this, fisheye distortion degrades ArUco pose accuracy.
DEFAULT_PERIPHERY_THRESHOLD = 0.6


@click.command(help="Calibrate the mapping ArUco tag pose in the SLAM map frame.")
@click.option(
    "-d",
    "--tag_detection",
    required=True,
    type=click.Path(exists=True),
    help="tag_detection.pkl from the mapping video.",
)
@click.option(
    "-c",
    "--csv_trajectory",
    required=True,
    type=click.Path(exists=True),
    help="camera_trajectory.csv from SLAM re-localization on the mapping video.",
)
@click.option("-o", "--output", required=True, help="Output tx_slam_tag.json path.")
@click.option(
    "-tid",
    "--tag_id",
    type=int,
    default=13,
    show_default=True,
    help="ArUco tag ID used as the mapping landmark.",
)
@click.option(
    "-k",
    "--keyframe_only",
    is_flag=True,
    default=False,
    help="Only use keyframe poses from the trajectory.",
)
@click.option(
    "-p",
    "--periphery_threshold",
    type=float,
    default=DEFAULT_PERIPHERY_THRESHOLD,
    show_default=True,
    help="Max distance from image center (fraction of half-height) for tag detections.",
)
def main(
    tag_detection, csv_trajectory, output, tag_id, keyframe_only, periphery_threshold
):
    """Calibrate the mapping ArUco tag pose in the SLAM map coordinate frame.

    :param tag_detection: Path to tag_detection.pkl from the mapping video.
    :param csv_trajectory: Path to camera_trajectory.csv from SLAM re-localization.
        Prefer camera_trajectory.csv over mapping_camera_trajectory.csv — it is
        produced by re-localizing the mapping video against the finished map atlas
        and is significantly more accurate.
    :param output: Output path for tx_slam_tag.json.
    :param tag_id: ArUco tag ID used as the mapping landmark.
    :param keyframe_only: If set, restrict to keyframe poses only.
    :param periphery_threshold: Max distance from image center (fraction of half-height)
        beyond which detections are discarded due to fisheye distortion.
    """
    df = pd.read_csv(csv_trajectory)
    with open(tag_detection, "rb") as f:
        tag_detection_results = pickle.load(f)

    is_valid = ~df["is_lost"]
    if keyframe_only:
        is_valid &= df["is_keyframe"]

    cam_pose_timestamps = df["timestamp"].loc[is_valid].to_numpy()
    cam_pos = df[["x", "y", "z"]].loc[is_valid].to_numpy()
    cam_rot_quat_xyzw = df[["q_x", "q_y", "q_z", "q_w"]].loc[is_valid].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:, 3, 3] = 1
    cam_pose[:, :3, 3] = cam_pos
    cam_pose[:, :3, :3] = cam_rot.as_matrix()

    # Match each camera pose timestamp to the nearest detection frame by time.
    video_timestamps = np.array([x["time"] for x in tag_detection_results])
    insert_idxs = np.searchsorted(video_timestamps, cam_pose_timestamps, side="left")
    prev_idxs = np.clip(insert_idxs - 1, 0, len(video_timestamps) - 1)
    next_idxs = np.clip(insert_idxs, 0, len(video_timestamps) - 1)
    prev_diff = np.abs(video_timestamps[prev_idxs] - cam_pose_timestamps)
    next_diff = np.abs(video_timestamps[next_idxs] - cam_pose_timestamps)
    video_idxs = np.where(next_diff < prev_diff, next_idxs, prev_idxs)

    all_tx_slam_tag = []
    for cam_idx, video_idx in enumerate(video_idxs):
        tag_dict = tag_detection_results[video_idx]["tag_dict"]
        if tag_id not in tag_dict:
            continue

        tag = tag_dict[tag_id]
        tx_cam_tag = pose_to_mat(np.concatenate([tag["tvec"], tag["rvec"]]))
        tx_slam_cam = cam_pose[cam_idx]

        # Discard detections where the tag is too close or too far from the camera.
        dist_to_cam = np.linalg.norm(tx_cam_tag[:3, 3])
        if dist_to_cam < MIN_TAG_DIST_M or dist_to_cam > MAX_TAG_DIST_M:
            continue

        # Discard detections near the image periphery where fisheye distortion is high.
        tag_center_pix = tag["corners"].mean(axis=0)
        img_center = np.array(GOPRO_2_7K_RESOLUTION[::-1], dtype=np.float32) / 2
        if (
            np.linalg.norm(tag_center_pix - img_center) / img_center[1]
            > periphery_threshold
        ):
            continue

        all_tx_slam_tag.append(tx_slam_cam @ tx_cam_tag)

    if not all_tx_slam_tag:
        raise click.ClickException(
            f"Tag ID {tag_id} was not observed in any valid camera pose after filtering. "
            "Check that the correct tag_id is used and the tag is visible in the mapping video."
        )

    all_tx_slam_tag = np.array(all_tx_slam_tag)

    # Robust selection: keep the 90th-percentile inliers by distance to geometric
    # median, then pick the sample closest to the inlier mean.
    all_slam_tag_pos = all_tx_slam_tag[:, :3, 3]
    median = geometric_median(all_slam_tag_pos)
    dists = np.linalg.norm(all_tx_slam_tag[:, :3, 3] - median, axis=-1)
    inlier_mask = dists < np.quantile(dists, 0.9)
    inlier_pos = all_slam_tag_pos[inlier_mask]
    std = inlier_pos.std(axis=0)
    mean = inlier_pos.mean(axis=0)
    nn_idx = np.argmin(
        np.linalg.norm(all_tx_slam_tag[inlier_mask][:, :3, 3] - mean, axis=-1)
    )
    tx_slam_tag = all_tx_slam_tag[inlier_mask][nn_idx]

    logger.info(f"Tag position std (cm, within 90th-pct inliers): {std * 100}")

    pathlib.Path(output).write_text(
        json.dumps({"tx_slam_tag": tx_slam_tag.tolist()}, indent=2)
    )
    logger.info(f"Saved result to {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
