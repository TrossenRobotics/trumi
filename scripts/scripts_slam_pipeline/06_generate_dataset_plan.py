"""Generate a dataset plan (dataset_plan.pkl).

:Steps:
    1. Gather video metadata (fps, timestamps, camera serials) from all demo videos.
    2. Match overlapping videos from all cameras into demo episodes.
    3. Identify gripper hardware id per camera using ArUco tag detection results.
    4. Disambiguate gripper left/right assignment using SLAM poses.
    5. For each episode, align frame indices across cameras, load SLAM poses and
       gripper widths, then segment into valid sub-episodes.
    6. Save dataset_plan.pkl containing per-episode TCP poses, gripper widths,
       and raw video frame ranges.

:Usage:
    uv run python scripts/scripts_slam_pipeline/06_generate_dataset_plan.py -i <session_dir>
"""

import collections
import json
import logging
import math
import pathlib
import pickle

import av
import click
import numpy as np
import pandas as pd
from exiftool import ExifToolHelper
from scipy.spatial.transform import Rotation

from trumi.utils.cv_util import get_gripper_width
from trumi.utils.interpolation_util import (
    PoseInterpolator,
    get_gripper_calibration_interpolator,
    get_interp1d,
)
from trumi.utils.pose_util import mat_to_pose, pose_to_mat
from trumi.utils.timecode_util import mp4_get_start_datetime

logger = logging.getLogger(__name__)


def get_bool_segments(bool_seq):
    """Split a boolean sequence into contiguous same-value runs.

    :param bool_seq: 1-D boolean sequence to segment.
    :return: Tuple of (segments, segment_type) where segments is a list of
        slice objects and segment_type is a bool array indicating whether
        each segment is True or False.
    """
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = list()
    segment_type = list()
    for i in range(len(segment_bounds) - 1):
        start = segment_bounds[i]
        end = segment_bounds[i + 1]
        this_type = bool_seq[start]
        segments.append(slice(start, end))
        segment_type.append(this_type)
    segment_type = np.array(segment_type, dtype=bool)
    return segments, segment_type


def pose_interp_from_df(df, start_timestamp=0.0, tx_base_slam=None):
    """Build a PoseInterpolator from a camera_trajectory DataFrame.

    :param df: DataFrame with columns timestamp, x, y, z, q_x, q_y, q_z, q_w.
    :param start_timestamp: Wall-clock offset (s) added to the relative timestamps.
    :param tx_base_slam: Optional (4, 4) transform from SLAM map to the desired
        base frame. If None, poses are returned in the SLAM map frame.
    :return: PoseInterpolator mapping wall-clock time to 6-DOF pose in base frame.
    """
    timestamp_sec = df["timestamp"].to_numpy() + start_timestamp
    cam_pos = df[["x", "y", "z"]].to_numpy()
    cam_rot_quat_xyzw = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:, 3, 3] = 1
    cam_pose[:, :3, 3] = cam_pos
    cam_pose[:, :3, :3] = cam_rot.as_matrix()
    tx_slam_cam = cam_pose
    tx_base_cam = tx_slam_cam
    if tx_base_slam is not None:
        tx_base_cam = tx_base_slam @ tx_slam_cam
    pose_interp = PoseInterpolator(t=timestamp_sec, x=mat_to_pose(tx_base_cam))
    return pose_interp


def get_x_projection(tx_tag_this, tx_tag_other):
    """Compute the lateral projection of another camera's position relative to this one.

    Projects the displacement vector from this camera to the other onto the axis
    cross(this_forward, world_up). The sign is only meaningful comparatively:
    argsort ascending across cameras gives right-to-left ordering.

    :param tx_tag_this: (N, 4, 4) transform matrices for the reference camera.
    :param tx_tag_other: (N, 4, 4) transform matrices for the other camera.
    :return: (N,) projection values.
    """
    t_this_other = tx_tag_other[:, :3, 3] - tx_tag_this[:, :3, 3]
    v_this_forward = tx_tag_this[:, :3, 2]
    v_up = np.array([0.0, 0.0, 1.0])
    v_this_right = np.cross(v_this_forward, v_up)
    proj_other_right = np.sum(v_this_right * t_this_other, axis=-1)
    return proj_other_right


@click.command(help="Generate dataset_plan.pkl from a processed session directory.")
@click.option(
    "-i", "--input", required=True, help="Session directory (parent of demos/)."
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output path for dataset_plan.pkl. Defaults to <input>/dataset_plan.pkl.",
)
@click.option(
    "-to",
    "--tcp_offset",
    type=float,
    default=0.205,
    help="Distance (m) from gripper tip to camera mounting screw.",
)
@click.option(
    "-ts",
    "--tx_slam_tag",
    default=None,
    help="Path to tx_slam_tag.json. Defaults to <input>/demos/mapping_*/tx_slam_tag.json.",
)
@click.option(
    "-nz",
    "--nominal_z",
    type=float,
    default=0.072,
    help="Nominal Z distance (m) between gripper finger tags.",
)
@click.option(
    "-ml",
    "--min_episode_length",
    type=int,
    default=24,
    help="Minimum valid episode length in SLAM frames.",
)
@click.option(
    "--ignore_cameras",
    type=str,
    default=None,
    help="Comma-separated camera serials to exclude.",
)
@click.option(
    "--finger_tag_det_th",
    type=float,
    default=0.8,
    help="Min joint detection probability to classify a camera as a gripper camera.",
)
@click.option(
    "-sfs",
    "--slam_frame_stride",
    type=int,
    default=2,
    help="Frame subsampling stride used by SLAM and ArUco detection "
    "(e.g. 2 when raw video is 120fps and SLAM ran at 60fps).",
)
def main(
    input,
    output,
    tcp_offset,
    tx_slam_tag,
    nominal_z,
    min_episode_length,
    ignore_cameras,
    finger_tag_det_th,
    slam_frame_stride,
):
    """Generate dataset_plan.pkl from a processed session directory.

    :param input: Session directory (parent of demos/).
    :param output: Output path for dataset_plan.pkl.
    :param tcp_offset: Distance (m) from gripper tip to camera mounting screw.
    :param tx_slam_tag: Path to tx_slam_tag.json.
    :param nominal_z: Nominal Z distance (m) between gripper finger tags.
    :param min_episode_length: Minimum valid episode length in SLAM frames.
    :param ignore_cameras: Comma-separated camera serials to exclude.
    :param finger_tag_det_th: Min joint detection probability to classify a camera as a gripper camera.
    :param slam_frame_stride: Frame stride used by SLAM and ArUco (raw_fps / slam_fps).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- stage 0 ---
    # gather inputs
    input_path = pathlib.Path(input).expanduser().resolve()
    demos_dir = input_path.joinpath("demos")
    if output is None:
        output = input_path.joinpath("dataset_plan.pkl")
    else:
        output = pathlib.Path(output).expanduser().resolve()

    # tcp to camera transform
    # all unit in meters
    # y axis in camera frame
    cam_to_center_height = 0.086  # constant for UMI
    # optical center to mounting screw, positive is when optical center is in front of the mount
    cam_to_mount_offset = 0.01465  # constant for GoPro Hero 9,10,11,12,13
    cam_to_tip_offset = cam_to_mount_offset + tcp_offset

    pose_cam_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0, 0, 0])
    tx_cam_tcp = pose_to_mat(pose_cam_tcp)

    # SLAM map origin to table tag transform
    if tx_slam_tag is None:
        mapping_dirs = sorted(demos_dir.glob("mapping_*"))
        if not mapping_dirs:
            raise click.ClickException(f"No mapping_* directory found in {demos_dir}")
        tx_slam_tag_path = mapping_dirs[0] / "tx_slam_tag.json"
    else:
        tx_slam_tag_path = pathlib.Path(tx_slam_tag).expanduser().resolve()
    if not tx_slam_tag_path.is_file():
        raise click.ClickException(f"tx_slam_tag.json not found: {tx_slam_tag_path}")
    tx_slam_tag = np.array(json.loads(tx_slam_tag_path.read_text())["tx_slam_tag"])
    tx_tag_slam = np.linalg.inv(tx_slam_tag)

    # load gripper calibration
    gripper_id_gripper_cal_map = dict()
    cam_serial_gripper_cal_map = dict()

    with ExifToolHelper() as et:
        for gripper_cal_path in demos_dir.glob("gripper*/gripper_range.json"):
            mp4_path = gripper_cal_path.parent.joinpath("raw_video.mp4")
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta["QuickTime:CameraSerialNumber"]

            gripper_range_data = json.loads(gripper_cal_path.read_text())
            gripper_id = gripper_range_data["gripper_id"]
            max_width = gripper_range_data["max_width"]
            min_width = gripper_range_data["min_width"]
            gripper_cal_data = {
                "aruco_measured_width": [min_width, max_width],
                "aruco_actual_width": [min_width, max_width],
            }
            gripper_cal_interp = get_gripper_calibration_interpolator(
                **gripper_cal_data
            )
            gripper_id_gripper_cal_map[gripper_id] = gripper_cal_interp
            cam_serial_gripper_cal_map[cam_serial] = gripper_cal_interp

    # --- stage 1 ---
    # loop over all demo directory to extract video metadata
    # output: video_meta_df

    # find videos
    video_dirs = sorted([x.parent for x in demos_dir.glob("demo_*/raw_video.mp4")])

    # ignore camera
    ignore_cam_serials = set()
    if ignore_cameras is not None:
        serials = ignore_cameras.split(",")
        ignore_cam_serials = set(serials)

    fps = None
    rows = list()
    with ExifToolHelper() as et:
        for video_dir in video_dirs:
            mp4_path = video_dir.joinpath("raw_video.mp4")
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta["QuickTime:CameraSerialNumber"]
            start_date = mp4_get_start_datetime(str(mp4_path))
            start_timestamp = start_date.timestamp()

            if cam_serial in ignore_cam_serials:
                logger.info("Ignored %s", video_dir.name)
                continue

            csv_path = video_dir.joinpath("camera_trajectory.csv")
            if not csv_path.is_file():
                logger.info("Ignored %s, no camera_trajectory.csv", video_dir.name)
                continue

            pkl_path = video_dir.joinpath("tag_detection.pkl")
            if not pkl_path.is_file():
                logger.info("Ignored %s, no tag_detection.pkl", video_dir.name)
                continue

            with av.open(str(mp4_path), "r") as container:
                stream = container.streams.video[0]
                n_frames = stream.frames
                if fps is None:
                    fps = stream.average_rate
                else:
                    if fps != stream.average_rate:
                        raise click.ClickException(
                            f"Inconsistent fps: {float(fps)} vs {float(stream.average_rate)} in {video_dir.name}"
                        )
            duration_sec = float(n_frames / fps)
            end_timestamp = start_timestamp + duration_sec

            rows.append(
                {
                    "video_dir": video_dir,
                    "camera_serial": cam_serial,
                    "start_date": start_date,
                    "n_frames": n_frames,
                    "fps": fps,
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                }
            )
    if len(rows) == 0:
        raise click.ClickException("No valid videos found!")

    video_meta_df = pd.DataFrame(data=rows)

    # --- stage 2 ---
    # match videos into demos
    # output:
    # demo_data_list = {
    #     "video_idxs": [int],
    #     # calculating start/end frame requires gripper info, defer to later stage
    #     "start_timestamp": float,
    #     "end_timestamp": float
    # }
    # map serial to count
    serial_count = video_meta_df["camera_serial"].value_counts()
    logger.info("Found following cameras:\n%s", serial_count.to_string())
    n_cameras = len(serial_count)

    events = list()
    for vid_idx, row in video_meta_df.iterrows():
        events.append(
            {
                "vid_idx": vid_idx,
                "camera_serial": row["camera_serial"],
                "t": row["start_timestamp"],
                "is_start": True,
            }
        )
        events.append(
            {
                "vid_idx": vid_idx,
                "camera_serial": row["camera_serial"],
                "t": row["end_timestamp"],
                "is_start": False,
            }
        )
    events = sorted(events, key=lambda x: x["t"])

    demo_data_list = list()
    on_videos = set()
    on_cameras = set()
    used_videos = set()
    t_demo_start = None
    for i, event in enumerate(events):
        # update state based on event
        if event["is_start"]:
            on_videos.add(event["vid_idx"])
            on_cameras.add(event["camera_serial"])
        else:
            on_videos.remove(event["vid_idx"])
            on_cameras.remove(event["camera_serial"])
        if len(on_videos) != len(on_cameras):
            raise RuntimeError(
                f"on_videos/on_cameras length mismatch: {len(on_videos)} vs {len(on_cameras)}"
            )

        if len(on_cameras) == n_cameras:
            # start demo episode where all cameras are recording
            t_demo_start = event["t"]
        elif t_demo_start is not None:
            # demo already started, but one camera stopped
            # stopping episode
            if event["is_start"]:
                raise RuntimeError(
                    f"Unexpected start event while demo is running for camera {event['camera_serial']}"
                )

            t_start = t_demo_start
            t_end = event["t"]

            # undo state update to get full set of videos
            demo_vid_idxs = set(on_videos)
            demo_vid_idxs.add(event["vid_idx"])
            used_videos.update(demo_vid_idxs)

            demo_data_list.append(
                {
                    "video_idxs": sorted(demo_vid_idxs),
                    "start_timestamp": t_start,
                    "end_timestamp": t_end,
                }
            )
            t_demo_start = None
    unused_videos = set(video_meta_df.index) - used_videos
    for vid_idx in unused_videos:
        logger.warning(
            "Video %s unused in any demo",
            video_meta_df.loc[vid_idx]["video_dir"].name,
        )

    # --- stage 3 ---
    # identify gripper id (hardware) using aruco
    # output:
    # add video_meta_df['gripper_hardware_id'] column
    # cam_serial_gripper_hardware_id_map Dict[str, int]
    # finger_tag_det_th set via CLI parameter
    vid_idx_gripper_hardware_id_map = dict()
    cam_serial_gripper_ids_map = collections.defaultdict(list)
    for vid_idx, row in video_meta_df.iterrows():
        video_dir = row["video_dir"]
        pkl_path = video_dir.joinpath("tag_detection.pkl")
        if not pkl_path.is_file():
            vid_idx_gripper_hardware_id_map[vid_idx] = -1
            continue
        tag_data = pickle.loads(pkl_path.read_bytes())
        n_frames = len(tag_data)
        tag_counts = collections.defaultdict(lambda: 0)
        for frame in tag_data:
            for key in frame["tag_dict"].keys():
                tag_counts[key] += 1
        tag_stats = collections.defaultdict(lambda: 0.0)
        for k, v in tag_counts.items():
            tag_stats[k] = v / n_frames

        # classify gripper by tag
        # tag 0, 1 are reserved for gripper 0
        # tag 6, 7 are reserved for gripper 1
        if n_frames == 0 or len(tag_stats) == 0:
            cam_serial_gripper_ids_map[row["camera_serial"]].append(-1)
            vid_idx_gripper_hardware_id_map[vid_idx] = -1
            continue
        max_tag_id = np.max(list(tag_stats.keys()))
        tag_per_gripper = 6
        max_gripper_id = max_tag_id // tag_per_gripper

        gripper_prob_map = dict()
        for gripper_id in range(max_gripper_id + 1):
            left_id = gripper_id * tag_per_gripper
            right_id = left_id + 1
            left_prob = tag_stats[left_id]
            right_prob = tag_stats[right_id]
            gripper_prob = min(left_prob, right_prob)
            if gripper_prob <= 0:
                continue
            gripper_prob_map[gripper_id] = gripper_prob

        gripper_id_by_tag = -1
        if len(gripper_prob_map) > 0:
            gripper_probs = sorted(gripper_prob_map.items(), key=lambda x: x[-1])
            gripper_id = gripper_probs[-1][0]
            gripper_prob = gripper_probs[-1][1]
            if gripper_prob >= finger_tag_det_th:
                gripper_id_by_tag = gripper_id

        cam_serial_gripper_ids_map[row["camera_serial"]].append(gripper_id_by_tag)
        vid_idx_gripper_hardware_id_map[vid_idx] = gripper_id_by_tag

    # add column to video_meta_df for gripper hardware id
    series = pd.Series(
        data=list(vid_idx_gripper_hardware_id_map.values()),
        index=list(vid_idx_gripper_hardware_id_map.keys()),
    )
    video_meta_df["gripper_hardware_id"] = series

    cam_serial_gripper_hardware_id_map = dict()
    for cam_serial, gripper_ids in cam_serial_gripper_ids_map.items():
        counter = collections.Counter(gripper_ids)
        if len(counter) != 1:
            logger.warning(
                "Multiple gripper ids %s detected for camera serial %s",
                counter,
                cam_serial,
            )
        gripper_id = counter.most_common()[0][0]
        cam_serial_gripper_hardware_id_map[cam_serial] = gripper_id

    # TODO(abhichothani42): Simplify the logic using predefined left and right gripper cam data
    # --- stage 4 ---
    # disambiguiate gripper left/right
    # camera idx / robot idx convention:
    # from right (0) to left (1)
    # non gripper cameras are after (2,3,4..) sorted by serial number
    # output
    # cam_serial_cam_idx_map Dict[str,int]
    # video_meta_df add column "camera_idx" and "camera_idx_from_episode"

    n_gripper_cams = (
        np.array(list(cam_serial_gripper_hardware_id_map.values())) >= 0
    ).sum()

    if n_gripper_cams <= 0:
        # no gripper camera
        raise RuntimeError("No gripper camera detected!")

    # classify cam serials
    grip_cam_serials = list()
    other_cam_serials = list()
    for cs, gi in cam_serial_gripper_hardware_id_map.items():
        if gi >= 0:
            grip_cam_serials.append(cs)
        else:
            other_cam_serials.append(cs)

    # assign non-gripper camera index by ascending camera serial
    cam_serial_cam_idx_map = dict()
    for i, cs in enumerate(sorted(other_cam_serials)):
        cam_serial_cam_idx_map[cs] = len(grip_cam_serials) + i

    # disambiguiate gripper left/right at each demo episode
    cam_serial_right_to_left_idx_map = collections.defaultdict(list)
    vid_idx_cam_idx_map = np.full(len(video_meta_df), fill_value=-1, dtype=np.int32)
    for demo_idx, demo_data in enumerate(demo_data_list):
        video_idxs = demo_data["video_idxs"]
        start_timestamp = demo_data["start_timestamp"]
        end_timestamp = demo_data["end_timestamp"]

        # build pose interpolator for each gripper video
        cam_serials = list()
        gripper_vid_idxs = list()
        pose_interps = list()

        for vid_idx in video_idxs:
            row = video_meta_df.loc[vid_idx]
            if row.gripper_hardware_id < 0:
                # not gripper camera
                cam_serial = row["camera_serial"]
                if cam_serial in cam_serial_cam_idx_map:
                    vid_idx_cam_idx_map[vid_idx] = cam_serial_cam_idx_map[cam_serial]
                continue

            cam_serials.append(row["camera_serial"])
            gripper_vid_idxs.append(vid_idx)
            vid_dir = row["video_dir"]

            csv_path = vid_dir.joinpath("camera_trajectory.csv")
            if not csv_path.is_file():
                # no tracking data
                break

            csv_df = pd.read_csv(csv_path)

            if csv_df["is_lost"].sum() > 10:
                # drop episode if too many lost frames
                # unreliable tracking
                break

            if (~csv_df["is_lost"]).sum() < 60:
                break

            df = csv_df.loc[~csv_df["is_lost"]]
            pose_interp = pose_interp_from_df(
                df,
                start_timestamp=row["start_timestamp"],
                # build pose in tag frame (z-up)
                tx_base_slam=tx_tag_slam,
            )
            pose_interps.append(pose_interp)

        if len(pose_interps) != n_gripper_cams:
            # invalid episode
            logger.warning("Excluded demo %d from left/right disambiguation.", demo_idx)
            continue

        # calculate x-projection for each other
        n_samples = 100
        t_samples = np.linspace(start_timestamp, end_timestamp, n_samples)
        pose_samples = [pose_to_mat(interp(t_samples)) for interp in pose_interps]

        # heuristic
        # project other camera's position
        # to the cross product of this camera's z (forward) and global z (up)
        # which is the "right" of the camera
        # if positive, this means the other camera is on the "right" of this camera
        # similarly, the most negative camera is the right-most camera (all others are on the left)
        x_proj_avg = list()
        for i in range(len(pose_samples)):
            # general formulation, compatible with even >2 grippers
            this_proj_avg = list()
            for j in range(len(pose_samples)):
                # 0 if i == j
                # keep this for single gripper case
                this_proj_avg.append(
                    np.mean(
                        get_x_projection(
                            tx_tag_this=pose_samples[i], tx_tag_other=pose_samples[j]
                        )
                    )
                )
            this_proj_avg = np.mean(this_proj_avg)
            x_proj_avg.append(this_proj_avg)

        # right camera/gripper is 0, left is 1
        camera_right_to_left_idxs = np.argsort(x_proj_avg)

        for vid_idx, cam_serial, cam_right_idx in zip(
            gripper_vid_idxs, cam_serials, camera_right_to_left_idxs
        ):
            # save result for aggregation
            cam_serial_right_to_left_idx_map[cam_serial].append(cam_right_idx)
            # save result for per-episode assignment
            vid_idx_cam_idx_map[vid_idx] = cam_right_idx

    # assign most common cam index to each gripper camera
    for cs, cis in cam_serial_right_to_left_idx_map.items():
        count = collections.Counter(cis)
        this_cam_idx = count.most_common(1)[0][0]
        cam_serial_cam_idx_map[cs] = this_cam_idx

    # create columns
    camera_idx_series = video_meta_df["camera_serial"].map(cam_serial_cam_idx_map)
    camera_idx_from_episode_series = pd.Series(
        data=vid_idx_cam_idx_map, index=video_meta_df.index
    )

    # modify df
    video_meta_df["camera_idx"] = camera_idx_series
    video_meta_df["camera_idx_from_episode"] = camera_idx_from_episode_series

    rows = list()
    for cs, ci in cam_serial_cam_idx_map.items():
        rows.append(
            {
                "camera_idx": ci,
                "camera_serial": cs,
                "gripper_hw_idx": cam_serial_gripper_hardware_id_map[cs],
                "example_vid": video_meta_df.loc[video_meta_df["camera_serial"] == cs]
                .iloc[0]["video_dir"]
                .name,
            }
        )
    camera_serial_df = pd.DataFrame(data=rows)
    camera_serial_df.set_index("camera_idx", inplace=True)
    camera_serial_df.sort_index(inplace=True)
    logger.info(
        "Assigned camera_idx: right=0; left=1; non_gripper=2,3...\n%s",
        camera_serial_df.to_string(),
    )

    # --- stage 5 ---
    # generate dataset plan
    # output
    # all_plans = [{
    #     "episode_timestamps": np.ndarray,
    #     "grippers": [{
    #         "tcp_pose": np.ndarray,
    #         "gripper_width": np.ndarray
    #     }],
    #     "cameras": [{
    #         "video_path": str,
    #         "video_start_end": Tuple[int,int]
    #     }]
    # }]
    total_available_time = 0.0
    total_used_time = 0.0
    dropped_camera_count = collections.defaultdict(lambda: 0)
    n_dropped_demos = 0
    all_plans = list()
    for demo_idx, demo_data in enumerate(demo_data_list):
        video_idxs = demo_data["video_idxs"]
        start_timestamp = demo_data["start_timestamp"]
        end_timestamp = demo_data["end_timestamp"]
        total_available_time += end_timestamp - start_timestamp

        # select relevant video data
        demo_video_meta_df = video_meta_df.loc[video_idxs].copy()
        demo_video_meta_df.set_index("camera_idx", inplace=True)
        demo_video_meta_df.sort_index(inplace=True)

        # determine optimal alignment
        dt = None
        alignment_costs = list()
        for cam_idx, row in demo_video_meta_df.iterrows():
            dt = 1 / row["fps"]
            this_alignment_cost = list()
            for other_cam_idx, other_row in demo_video_meta_df.iterrows():
                # what's the delay for previous frame
                diff = other_row["start_timestamp"] - row["start_timestamp"]
                remainder = diff % dt
                this_alignment_cost.append(remainder)
            alignment_costs.append(this_alignment_cost)
        # first video in bundle
        align_cam_idx = np.argmin([sum(x) for x in alignment_costs])

        # rewrite start_timestamp to be integer multiple of dt
        align_video_start = demo_video_meta_df.loc[align_cam_idx]["start_timestamp"]
        start_timestamp += dt - ((start_timestamp - align_video_start) % dt)

        # dt is raw video frame interval (eg 1/120s), dt_slam is SLAM frame interval (eg 1/60s)
        dt_slam = dt * slam_frame_stride

        # cam_start_frame_idxs: raw-fps start frame per camera used in video_start_end output
        # cam_slam_frame_idxs:  slam-fps start frame per camera used for csv/pkl slicing
        cam_start_frame_idxs = list()
        cam_slam_frame_idxs = list()
        n_frames = int((end_timestamp - start_timestamp) / dt_slam)
        for cam_idx, row in demo_video_meta_df.iterrows():
            video_start_frame = math.ceil(
                (start_timestamp - row["start_timestamp"]) / dt
            )
            video_n_frames = (
                math.floor((row["end_timestamp"] - start_timestamp) / dt_slam) - 1
            )
            if video_start_frame < 0:
                video_n_frames += video_start_frame // slam_frame_stride
                video_start_frame = 0
            # round up to slam_frame_stride boundary so SLAM/ArUco indices stay in sync with raw frames
            if video_start_frame % slam_frame_stride != 0:
                video_start_frame += slam_frame_stride - (
                    video_start_frame % slam_frame_stride
                )
            slam_start_frame = video_start_frame // slam_frame_stride
            cam_start_frame_idxs.append(video_start_frame)
            cam_slam_frame_idxs.append(slam_start_frame)
            n_frames = min(n_frames, video_n_frames)
        demo_timestamps = np.arange(n_frames) * dt_slam + start_timestamp

        # load pose and gripper data for each video
        # determine valid frames for each video
        all_cam_poses = list()
        all_gripper_widths = list()
        all_is_valid = list()

        for cam_idx, row in demo_video_meta_df.iterrows():
            if cam_idx >= n_gripper_cams:
                # not gripper camera
                continue

            start_frame_idx = cam_slam_frame_idxs[cam_idx]
            video_dir = row["video_dir"]

            # load check data
            check_path = video_dir.joinpath("check_result.txt")
            if check_path.is_file():
                if not check_path.open("r").read().startswith("true"):
                    logger.info(
                        "Skipping %s, manually filtered with check_result.txt!=true",
                        video_dir.name,
                    )
                    continue

            # load SLAM data
            csv_path = video_dir.joinpath("camera_trajectory.csv")
            if not csv_path.is_file():
                logger.info("Skipping %s, no camera_trajectory.csv.", video_dir.name)
                dropped_camera_count[row["camera_serial"]] += 1
                continue

            csv_df = pd.read_csv(csv_path)
            # select aligned frames
            df = csv_df.iloc[start_frame_idx : start_frame_idx + n_frames]
            is_tracked = (~df["is_lost"]).to_numpy()

            # basic filtering to remove bad tracking
            n_frames_lost = (~is_tracked).sum()
            if n_frames_lost > 10:
                logger.info(
                    "Skipping %s, %d frames are lost.", video_dir.name, n_frames_lost
                )
                dropped_camera_count[row["camera_serial"]] += 1
                continue

            n_frames_valid = is_tracked.sum()
            if n_frames_valid < 60:
                logger.info(
                    "Skipping %s, only %d frames are valid.",
                    video_dir.name,
                    n_frames_valid,
                )
                dropped_camera_count[row["camera_serial"]] += 1
                continue

            # load camera pose
            df = df.copy()
            df.loc[df["is_lost"], ["q_x", "q_y", "q_z"]] = 0.0
            df.loc[df["is_lost"], "q_w"] = 1.0
            cam_pos = df[["x", "y", "z"]].to_numpy()
            cam_rot_quat_xyzw = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy()
            cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
            cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
            cam_pose[:, 3, 3] = 1
            cam_pose[:, :3, 3] = cam_pos
            cam_pose[:, :3, :3] = cam_rot.as_matrix()
            tx_slam_cam = cam_pose
            tx_tag_cam = tx_tag_slam @ tx_slam_cam

            # TODO(abhichothani42): handle optional robot cal based filtering
            is_step_valid = is_tracked.copy()

            # get gripper data
            pkl_path = video_dir.joinpath("tag_detection.pkl")
            if not pkl_path.is_file():
                logger.info("Skipping %s, no tag_detection.pkl.", video_dir.name)
                dropped_camera_count[row["camera_serial"]] += 1
                continue

            tag_detection_results = pickle.loads(pkl_path.read_bytes())[
                start_frame_idx : start_frame_idx + n_frames
            ]

            # one item per frame
            video_timestamps = np.array([x["time"] for x in tag_detection_results])

            if len(df) != len(video_timestamps):
                logger.info("Skipping %s, video csv length mismatch.", video_dir.name)
                continue

            # get gripper action
            ghi = row["gripper_hardware_id"]
            if ghi < 0:
                logger.info(
                    "Skipping %s, invalid gripper hardware id %d", video_dir.name, ghi
                )
                dropped_camera_count[row["camera_serial"]] += 1
                continue

            left_id = 6 * ghi
            right_id = left_id + 1

            gripper_cal_interp = None
            if ghi in gripper_id_gripper_cal_map:
                gripper_cal_interp = gripper_id_gripper_cal_map[ghi]
            elif row["camera_serial"] in cam_serial_gripper_cal_map:
                gripper_cal_interp = cam_serial_gripper_cal_map[row["camera_serial"]]
                logger.warning(
                    "Gripper id %d not found in gripper calibrations %s. Falling back to camera serial map.",
                    ghi,
                    list(gripper_id_gripper_cal_map.keys()),
                )
            else:
                raise RuntimeError("Gripper calibration not found.")

            gripper_timestamps = list()
            gripper_widths = list()
            for td in tag_detection_results:
                width = get_gripper_width(
                    td["tag_dict"],
                    left_id=left_id,
                    right_id=right_id,
                    nominal_z=nominal_z,
                )
                if width is not None:
                    gripper_timestamps.append(td["time"])
                    gripper_widths.append(gripper_cal_interp(width))
            if len(gripper_widths) < 2:
                logger.warning(
                    "Skipping %s: fewer than 2 gripper tag detections, cannot interpolate.",
                    video_dir.name,
                )
                dropped_camera_count[row["camera_serial"]] += 1
                continue
            gripper_interp = get_interp1d(gripper_timestamps, gripper_widths)

            gripper_det_ratio = len(gripper_widths) / len(tag_detection_results)
            if gripper_det_ratio < 0.9:
                logger.warning(
                    "%s: only %.0f%% of gripper tags detected.",
                    video_dir.name,
                    gripper_det_ratio * 100,
                )

            this_gripper_widths = gripper_interp(video_timestamps)

            # transform to tcp frame
            tx_tag_tcp = tx_tag_cam @ tx_cam_tcp
            pose_tag_tcp = mat_to_pose(tx_tag_tcp)

            # output value
            if len(pose_tag_tcp) != n_frames:
                raise RuntimeError(
                    f"pose length {len(pose_tag_tcp)} != n_frames {n_frames} in {video_dir.name}"
                )
            if len(this_gripper_widths) != n_frames:
                raise RuntimeError(
                    f"gripper_width length {len(this_gripper_widths)} != n_frames {n_frames} in {video_dir.name}"
                )
            if len(is_step_valid) != n_frames:
                raise RuntimeError(
                    f"is_step_valid length {len(is_step_valid)} != n_frames {n_frames} in {video_dir.name}"
                )
            all_cam_poses.append(pose_tag_tcp)
            all_gripper_widths.append(this_gripper_widths)
            all_is_valid.append(is_step_valid)

        if len(all_cam_poses) != n_gripper_cams:
            logger.info("Skipped demo %d.", demo_idx)
            n_dropped_demos += 1
            continue

        # aggregate valid result
        all_is_valid = np.array(all_is_valid)
        is_step_valid = np.all(all_is_valid, axis=0)

        # generate episode start and end pose for each gripper
        if not np.any(is_step_valid):
            logger.info("Skipped demo %d, no valid steps.", demo_idx)
            n_dropped_demos += 1
            continue
        first_valid_step = np.nonzero(is_step_valid)[0][0]
        last_valid_step = np.nonzero(is_step_valid)[0][-1]
        demo_start_poses = list()
        demo_end_poses = list()
        for cam_idx in range(len(all_cam_poses)):
            cam_poses = all_cam_poses[cam_idx]
            demo_start_poses.append(cam_poses[first_valid_step])
            demo_end_poses.append(cam_poses[last_valid_step])

        # determine episode segmentation
        # remove valid segments that are too short
        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for s, is_valid_segment in zip(segment_slices, segment_type):
            start = s.start
            end = s.stop
            if not is_valid_segment:
                continue
            if (end - start) < min_episode_length:
                is_step_valid[start:end] = False

        # finally, generate one episode for each valid segment
        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for s, is_valid in zip(segment_slices, segment_type):
            if not is_valid:
                continue
            start = s.start
            end = s.stop

            total_used_time += float((end - start) * dt_slam)

            grippers = list()
            cameras = list()
            for cam_idx, row in demo_video_meta_df.iterrows():
                if cam_idx < n_gripper_cams:
                    pose_tag_tcp = all_cam_poses[cam_idx][start:end]

                    # gripper cam
                    grippers.append(
                        {
                            "tcp_pose": pose_tag_tcp,
                            "gripper_width": all_gripper_widths[cam_idx][start:end],
                            "demo_start_pose": demo_start_poses[cam_idx],
                            "demo_end_pose": demo_end_poses[cam_idx],
                        }
                    )
                # all cams
                video_dir = row["video_dir"]
                vid_start_frame = cam_start_frame_idxs[cam_idx]
                cameras.append(
                    {
                        "video_path": str(
                            video_dir.joinpath("raw_video.mp4").relative_to(
                                video_dir.parent
                            )
                        ),
                        "video_start_end": (
                            start * slam_frame_stride + vid_start_frame,
                            end * slam_frame_stride + vid_start_frame,
                        ),
                    }
                )

            all_plans.append(
                {
                    "episode_timestamps": demo_timestamps[start:end],
                    "grippers": grippers,
                    "cameras": cameras,
                }
            )

    used_ratio = total_used_time / total_available_time
    logger.info("%d%% of raw data are used.", int(used_ratio * 100))

    if dropped_camera_count:
        logger.info("Dropped camera counts: %s", dict(dropped_camera_count))
    logger.info("Dropped demos: %d", n_dropped_demos)

    output.write_bytes(pickle.dumps(all_plans))
    logger.info("Saved dataset plan (%d episodes) to %s", len(all_plans), output)


if __name__ == "__main__":
    main()
