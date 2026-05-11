"""Generate an MCAP dataset from SLAM pipeline outputs.

:Steps:
    1. Load dataset_plan.pkl from each input directory and extract
       robot state (eef pose, gripper width, demo start/end poses) for
       every episode.  For each camera, load imu_data.json for per-frame
       timestamps and IMU telemetry.

    2. For every raw video referenced by the plan, stream all frames in
       the episode range.  Per frame, optionally:

       a. Inpaint detected ArUco tags (--inpaint_aruco).
       b. Mask out the gripper region (--mask_gripper).
       c. Mask out mirror regions (--no_mirror).
       d. Resize the image (--out_res).
       e. Undistort via fisheye rectification (--out_fov).
       f. Apply mirror-swap augmentation (--mirror_swap).
       g. JPEG-compress the image.

    3. Write one MCAP file per episode. Camera frames, EEF/gripper
       state, and IMU samples are merged in wall-clock order.

:MCAP conventions:
    - One .mcap file per episode (episode_NNNNNN.mcap)
    - Profile: trossen
    - JSON message encoding with jsonschema schemas
    - foxglove.CompressedImage for camera frames (JPEG)
    - Configurable chunk compression (--compression)
    - MCAP metadata record trumi_recording per episode

:Topics (per episode file):
    robot{N}/eef/state             — {pos:[x,y,z], rot_axis_angle:[ax,ay,az]}
    robot{N}/gripper/state         — {width: float}
    robot{N}/eef_demo_start/state  — {pos:[x,y,z], rot_axis_angle:[ax,ay,az]}
    robot{N}/eef_demo_end/state    — {pos:[x,y,z], rot_axis_angle:[ax,ay,az]}
    robot{N}/imu                   — {timestamp, linear_acceleration, angular_velocity}
    /cameras/camera{N}/image       — foxglove.CompressedImage (JPEG)

:Output structure:
    <output_dir>/
        episode_000000.mcap
        episode_000001.mcap
        ...

:Usage:
    uv run python scripts/scripts_slam_pipeline/07_generate_mcap_dataset.py \\
        -o <session_dir>/dataset_mcap \\
        <session_dir>
"""

import base64
import concurrent.futures
import heapq
import json
import logging
import multiprocessing
import pathlib
import pickle
import types

import av
import click
import cv2
import numpy as np
from mcap.writer import CompressionType, Writer
from tqdm import tqdm

from trumi.utils.cv_util import (
    FisheyeRectConverter,
    draw_predefined_mask,
    get_image_transform,
    inpaint_tag,
    parse_fisheye_intrinsics,
)
from trumi.utils.timecode_util import mp4_get_start_datetime

logger = logging.getLogger(__name__)


# JSON schemas
_SCHEMAS = {
    "trumi.msg.EefState": {
        "type": "object",
        "properties": {
            "pos": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
            "rot_axis_angle": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["pos", "rot_axis_angle"],
    },
    "trumi.msg.GripperState": {
        "type": "object",
        "properties": {
            "width": {"type": "number"},
        },
        "required": ["width"],
    },
    "trumi.msg.ImuSample": {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "object",
                "properties": {
                    "sec": {"type": "integer"},
                    "nsec": {"type": "integer"},
                },
                "required": ["sec", "nsec"],
            },
            "linear_acceleration": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
            "angular_velocity": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["timestamp", "linear_acceleration", "angular_velocity"],
    },
    "foxglove.CompressedImage": {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "object",
                "properties": {
                    "sec": {"type": "integer"},
                    "nsec": {"type": "integer"},
                },
                "required": ["sec", "nsec"],
            },
            "frame_id": {"type": "string"},
            "format": {"type": "string"},
            "data": {"type": "string", "contentEncoding": "base64"},
        },
        "required": ["timestamp", "frame_id", "format", "data"],
    },
}

_COMPRESSION_MAP = {
    "zstd": CompressionType.ZSTD,
    "lz4": CompressionType.LZ4,
    "none": CompressionType.NONE,
}


def _encode_eef_state(pos, rot):
    """Encode eef position (3,) and rotation (3,) as JSON bytes."""
    return json.dumps(
        {
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rot_axis_angle": [float(rot[0]), float(rot[1]), float(rot[2])],
        }
    ).encode()


def _encode_gripper_state(width):
    """Encode gripper width as JSON bytes."""
    return json.dumps({"width": float(width)}).encode()


def _encode_imu_sample(t_ns, linear_acceleration, angular_velocity):
    """Encode a trumi.msg.ImuSample as JSON bytes."""
    sec = int(t_ns // 1_000_000_000)
    nsec = int(t_ns % 1_000_000_000)
    return json.dumps(
        {
            "timestamp": {"sec": sec, "nsec": nsec},
            "linear_acceleration": [float(x) for x in linear_acceleration],
            "angular_velocity": [float(x) for x in angular_velocity],
        }
    ).encode()


def _encode_compressed_image(img_rgb, t_ns, frame_id, quality=95):
    """Encode an RGB image as a foxglove.CompressedImage JSON message."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    sec = int(t_ns // 1_000_000_000)
    nsec = int(t_ns % 1_000_000_000)
    return json.dumps(
        {
            "timestamp": {"sec": sec, "nsec": nsec},
            "frame_id": frame_id,
            "format": "jpeg",
            "data": b64,
        }
    ).encode()


def _camera_frame_gen(source, cfg):
    """Yield (t_wall_ns, channel_key, encoded_bytes) for every raw frame."""
    cam_id = source["camera_idx"]
    video_path = source["video_path"]
    frame_start = source["frame_start"]
    frame_end = source["frame_end"]
    frame_wall_times_s = source["frame_wall_times_s"]
    channel_key = f"camera{cam_id}"

    aruco_detections = None
    if cfg.inpaint_aruco:
        pkl_path = pathlib.Path(video_path).parent / "tag_detection.pkl"
        if not pkl_path.is_file():
            raise click.ClickException(
                f"tag_detection.pkl not found (required for --inpaint_aruco): {pkl_path}"
            )
        aruco_detections = pickle.loads(pkl_path.read_bytes())

    with av.open(video_path) as container:
        in_stream = container.streams.video[0]
        in_stream.thread_count = 1

        for frame_idx, frame in enumerate(container.decode(in_stream)):
            if frame_idx < frame_start:
                continue
            if frame_idx >= frame_end:
                break

            img = frame.to_ndarray(format="rgb24")
            ep_frame_idx = frame_idx - frame_start

            if cfg.inpaint_aruco and aruco_detections is not None:
                slam_idx = frame_idx // cfg.slam_frame_stride
                if slam_idx < len(aruco_detections):
                    for tag in aruco_detections[slam_idx]["tag_dict"].values():
                        img = inpaint_tag(img, tag["corners"])

            if cfg.mask_gripper:
                img = draw_predefined_mask(
                    img, color=(0, 0, 0), mirror=False, gripper=True, finger=False
                )

            if cfg.no_mirror:
                img = draw_predefined_mask(
                    img, color=(0, 0, 0), mirror=True, gripper=False, finger=False
                )

            if cfg.fisheye_converter is not None:
                img = cfg.fisheye_converter.forward(img)
            elif cfg.resize_tf is not None:
                img = cfg.resize_tf(img)

            if cfg.mirror_swap and cfg.is_mirror is not None:
                img[cfg.is_mirror] = img[:, ::-1, :][cfg.is_mirror]

            t_wall_ns = int(frame_wall_times_s[ep_frame_idx] * 1e9)
            yield (
                t_wall_ns,
                channel_key,
                _encode_compressed_image(
                    img, t_wall_ns, f"camera{cam_id}", cfg.jpeg_quality
                ),
            )


def _write_episode(ep_idx, episode, cfg):
    """Write a single episode MCAP file."""
    timestamps = episode["timestamps"]
    robot_state = episode["robot_state"]
    video_sources = episode["video_sources"]
    ep_len = len(timestamps)

    episode_path = cfg.output_dir / f"episode_{ep_idx:06d}.mcap"

    with open(episode_path, "wb") as f:
        writer = Writer(f, compression=cfg.compression_type)
        writer.start(profile="trossen", library="trumi")

        # Register schemas
        schema_ids = {}
        for schema_name, schema_def in _SCHEMAS.items():
            schema_ids[schema_name] = writer.register_schema(
                name=schema_name,
                encoding="jsonschema",
                data=json.dumps(schema_def).encode(),
            )

        # Register channels
        channel_ids = {}

        for gripper_id in range(cfg.n_grippers):
            channel_ids[f"robot{gripper_id}_eef"] = writer.register_channel(
                schema_id=schema_ids["trumi.msg.EefState"],
                topic=f"robot{gripper_id}/eef/state",
                message_encoding="json",
            )
            channel_ids[f"robot{gripper_id}_gripper"] = writer.register_channel(
                schema_id=schema_ids["trumi.msg.GripperState"],
                topic=f"robot{gripper_id}/gripper/state",
                message_encoding="json",
            )
            channel_ids[f"robot{gripper_id}_demo_start"] = writer.register_channel(
                schema_id=schema_ids["trumi.msg.EefState"],
                topic=f"robot{gripper_id}/eef_demo_start/state",
                message_encoding="json",
            )
            channel_ids[f"robot{gripper_id}_demo_end"] = writer.register_channel(
                schema_id=schema_ids["trumi.msg.EefState"],
                topic=f"robot{gripper_id}/eef_demo_end/state",
                message_encoding="json",
            )
            channel_ids[f"robot{gripper_id}_imu"] = writer.register_channel(
                schema_id=schema_ids["trumi.msg.ImuSample"],
                topic=f"robot{gripper_id}/imu",
                message_encoding="json",
            )

        for cam_id in range(cfg.n_cameras):
            channel_ids[f"camera{cam_id}"] = writer.register_channel(
                schema_id=schema_ids["foxglove.CompressedImage"],
                topic=f"/cameras/camera{cam_id}/image",
                message_encoding="json",
                metadata={"stream_type": "color"},
            )

        # Write MCAP metadata record
        writer.add_metadata(
            name="trumi_recording",
            data={
                "episode_index": str(ep_idx),
                "episode_length": str(ep_len),
                "n_grippers": str(cfg.n_grippers),
                "n_cameras": str(cfg.n_cameras),
                "image_width": str(cfg.image_w),
                "image_height": str(cfg.image_h),
                "video_fps": str(cfg.video_fps),
                "slam_frame_stride": str(cfg.slam_frame_stride),
                "compression": cfg.compression,
                "jpeg_quality": str(cfg.jpeg_quality),
            },
        )

        # Build robot state + IMU messages as (t_ns, channel_key, data) tuples
        state_msgs = []

        # Write demo_start_pose and demo_end_pose once at t0
        t0_ns = int(timestamps[0] * 1e9)
        for gripper_id in range(cfg.n_grippers):
            start_pose = robot_state[f"robot{gripper_id}_demo_start_pose"]
            end_pose = robot_state[f"robot{gripper_id}_demo_end_pose"]
            state_msgs.append(
                (
                    t0_ns,
                    f"robot{gripper_id}_demo_start",
                    _encode_eef_state(start_pose[:3], start_pose[3:]),
                )
            )
            state_msgs.append(
                (
                    t0_ns,
                    f"robot{gripper_id}_demo_end",
                    _encode_eef_state(end_pose[:3], end_pose[3:]),
                )
            )

        # Per-step EEF and gripper data
        for step in range(ep_len):
            t_ns = int(timestamps[step] * 1e9)
            for gripper_id in range(cfg.n_grippers):
                state_msgs.append(
                    (
                        t_ns,
                        f"robot{gripper_id}_eef",
                        _encode_eef_state(
                            robot_state[f"robot{gripper_id}_eef_pos"][step],
                            robot_state[f"robot{gripper_id}_eef_rot"][step],
                        ),
                    )
                )
                state_msgs.append(
                    (
                        t_ns,
                        f"robot{gripper_id}_gripper",
                        _encode_gripper_state(
                            robot_state[f"robot{gripper_id}_gripper_width"][step]
                        ),
                    )
                )

        # IMU samples from each gripper camera
        for source in video_sources:
            imu = source["imu_samples"]
            if imu is None:
                continue
            gripper_id = source["camera_idx"]
            for j, t_imu_ns in enumerate(imu["wall_ns"]):
                state_msgs.append(
                    (
                        int(t_imu_ns),
                        f"robot{gripper_id}_imu",
                        _encode_imu_sample(
                            int(t_imu_ns), imu["accl"][j], imu["gyro"][j]
                        ),
                    )
                )

        state_msgs.sort(key=lambda m: m[0])

        # Merge state messages with per-camera frame streams in wall-clock order
        cam_gens = [_camera_frame_gen(s, cfg) for s in video_sources]
        all_msgs = heapq.merge(iter(state_msgs), *cam_gens, key=lambda m: m[0])

        for seq, (t_ns, channel_key, data) in enumerate(all_msgs):
            writer.add_message(
                channel_id=channel_ids[channel_key],
                log_time=t_ns,
                publish_time=t_ns,
                sequence=seq,
                data=data,
            )

        writer.finish()

    logger.debug("Wrote episode %d (%d steps) to %s", ep_idx, ep_len, episode_path)


@click.command(help="Generate an MCAP dataset from SLAM pipeline outputs.")
@click.argument("input_dirs", nargs=-1, required=True)
@click.option(
    "-o", "--output", required=True, help="Output directory for episode .mcap files."
)
@click.option(
    "-or",
    "--out_res",
    type=str,
    default=None,
    help="Output image resolution as W,H. Omit to keep native resolution.",
)
@click.option(
    "-of",
    "--out_fov",
    type=float,
    default=None,
    help="Vertical FOV (degrees) for fisheye rectification. Requires --out_res and --camera_intrinsics.",
)
@click.option(
    "-jq",
    "--jpeg_quality",
    type=int,
    default=95,
    show_default=True,
    help="JPEG quality for image compression (1-100).",
)
@click.option(
    "--inpaint_aruco",
    is_flag=True,
    default=False,
    help="Inpaint detected ArUco tags in camera frames.",
)
@click.option(
    "--mask_gripper",
    is_flag=True,
    default=False,
    help="Mask out the predefined gripper region from camera frames.",
)
@click.option(
    "-nm",
    "--no_mirror",
    is_flag=True,
    default=False,
    help="Mask out mirror regions from observations.",
)
@click.option(
    "-ms",
    "--mirror_swap",
    is_flag=True,
    default=False,
    help="Apply mirror-swap augmentation.",
)
@click.option(
    "-ci",
    "--camera_intrinsics",
    default=None,
    help="Path to fisheye intrinsics JSON. Required when --out_fov is set.",
)
@click.option(
    "--compression",
    type=click.Choice(["zstd", "lz4", "none"], case_sensitive=False),
    default="zstd",
    show_default=True,
    help="MCAP chunk compression algorithm.",
)
@click.option(
    "-n",
    "--num_workers",
    type=int,
    default=None,
    help="Number of parallel episode-writing threads. Defaults to half CPU count.",
)
@click.option(
    "-sfs",
    "--slam_frame_stride",
    type=int,
    default=2,
    help="Frame stride used by SLAM and ArUco detection.",
)
def main(
    input_dirs,
    output,
    out_res,
    out_fov,
    jpeg_quality,
    inpaint_aruco,
    mask_gripper,
    camera_intrinsics,
    no_mirror,
    mirror_swap,
    compression,
    num_workers,
    slam_frame_stride,
):
    """Generate an MCAP dataset from SLAM pipeline outputs.

    :param input_dirs: One or more input directories containing dataset_plan.pkl.
    :param output: Output directory for episode .mcap files.
    :param out_res: Output image resolution as 'W,H'. None keeps native resolution.
    :param out_fov: Vertical FOV for fisheye rectification (degrees).
    :param jpeg_quality: JPEG quality for image compression.
    :param inpaint_aruco: Inpaint detected ArUco tags in each frame.
    :param mask_gripper: Mask the gripper region with black pixels.
    :param camera_intrinsics: Path to fisheye intrinsics JSON.
    :param no_mirror: Mask out mirror regions from observations.
    :param mirror_swap: Apply mirror-swap augmentation.
    :param compression: MCAP chunk compression algorithm.
    :param num_workers: Number of parallel episode-writing threads.
    :param slam_frame_stride: Frame stride matching SLAM and ArUco detection rate.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_path = pathlib.Path(output).expanduser().resolve()
    if output_path.is_dir() and any(output_path.glob("episode_*.mcap")):
        if not click.confirm(
            f"Output directory {output_path} already contains episode files. Overwrite?"
        ):
            raise click.Abort()
        for stale in output_path.glob("episode_*.mcap"):
            stale.unlink()

    out_res_tuple = None
    if out_res is not None:
        out_res_tuple = tuple(int(x) for x in out_res.split(","))

    compression_type = _COMPRESSION_MAP[compression.lower()]

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)
    cv2.setNumThreads(1)

    fisheye_converter = None
    if out_fov is not None:
        if camera_intrinsics is None:
            raise click.UsageError(
                "--camera_intrinsics is required when --out_fov is set."
            )
        if out_res_tuple is None:
            raise click.UsageError("--out_res is required when --out_fov is set.")
        intr_path = pathlib.Path(camera_intrinsics).expanduser().resolve()
        if not intr_path.is_file():
            raise click.ClickException(f"Camera intrinsics not found: {intr_path}")
        opencv_intr_dict = parse_fisheye_intrinsics(json.loads(intr_path.read_text()))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict, out_size=out_res_tuple, out_fov=out_fov
        )

    # Phase 1: Collect all episodes
    all_episodes = []
    n_grippers = None
    n_cameras = None

    for ipath in input_dirs:
        ipath = pathlib.Path(ipath).expanduser().resolve()
        demos_path = ipath.joinpath("demos")
        plan_path = ipath.joinpath("dataset_plan.pkl")
        if not plan_path.is_file():
            logger.info("Skipping %s: no dataset_plan.pkl", ipath.name)
            continue

        plan = pickle.loads(plan_path.read_bytes())

        for plan_episode in plan:
            grippers = plan_episode["grippers"]
            cameras = plan_episode["cameras"]

            if n_grippers is None:
                n_grippers = len(grippers)
            elif n_grippers != len(grippers):
                raise click.ClickException(
                    f"Inconsistent gripper count: expected {n_grippers}, got {len(grippers)}."
                )
            if n_cameras is None:
                n_cameras = len(cameras)
            elif n_cameras != len(cameras):
                raise click.ClickException(
                    f"Inconsistent camera count: expected {n_cameras}, got {len(cameras)}."
                )

            episode_timestamps = plan_episode["episode_timestamps"]
            ep_len = len(episode_timestamps)

            # Build robot state data
            robot_state = {}
            for gripper_id, gripper in enumerate(grippers):
                eef_pose = gripper["tcp_pose"]
                eef_pos = eef_pose[..., :3].astype(np.float32)
                eef_rot = eef_pose[..., 3:].astype(np.float32)
                gripper_widths = gripper["gripper_width"].astype(np.float32)
                demo_start_pose = gripper["demo_start_pose"].astype(np.float32)
                demo_end_pose = gripper["demo_end_pose"].astype(np.float32)

                robot_state[f"robot{gripper_id}_eef_pos"] = eef_pos
                robot_state[f"robot{gripper_id}_eef_rot"] = eef_rot
                robot_state[f"robot{gripper_id}_gripper_width"] = gripper_widths
                robot_state[f"robot{gripper_id}_demo_start_pose"] = demo_start_pose
                robot_state[f"robot{gripper_id}_demo_end_pose"] = demo_end_pose

            for key, arr in robot_state.items():
                if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
                    continue  # these are single poses, not per-step
                if arr.shape[0] != ep_len:
                    raise RuntimeError(
                        f"Robot state '{key}' length {arr.shape[0]} != episode length {ep_len}"
                    )

            # Build video source info per camera
            video_sources = []
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera["video_path"]
                video_path = demos_path.joinpath(video_path_rel).absolute()
                if not video_path.is_file():
                    raise click.ClickException(f"Video file not found: {video_path}")

                frame_start, frame_end = camera["video_start_end"]
                n_frames_expected = (frame_end - frame_start) // slam_frame_stride
                if n_frames_expected != ep_len:
                    raise click.ClickException(
                        f"Frame count mismatch for camera {cam_id}: "
                        f"expected {ep_len}, got {n_frames_expected}."
                    )

                # Wall-clock start from ExifTool (Unix epoch seconds)
                video_wall_start = mp4_get_start_datetime(str(video_path)).timestamp()

                # GPMF per-frame offsets from video start
                imu_json_path = video_path.parent / "imu_data.json"
                if not imu_json_path.is_file():
                    raise click.ClickException(
                        f"imu_data.json not found: {imu_json_path}"
                    )
                imu_data = json.loads(imu_json_path.read_bytes())
                gpmf_frame_offsets_s = np.array(imu_data["img_timestamps_s"])

                # Wall-clock time for every raw frame in the episode window
                frame_wall_times_s = (
                    video_wall_start + gpmf_frame_offsets_s[frame_start:frame_end]
                )

                # IMU data for gripper cameras only (cam_id < n_grippers)
                imu_samples = None
                if cam_id < n_grippers:
                    accl_offsets_s = np.array(imu_data["ACCL"]["timestamps_s"])
                    accl_xyz = np.array(imu_data["ACCL"]["data"])
                    gyro_xyz = np.array(imu_data["GYRO"]["data"])

                    t_start_s = gpmf_frame_offsets_s[frame_start]
                    t_end_s = gpmf_frame_offsets_s[frame_end - 1]
                    mask = (accl_offsets_s >= t_start_s) & (accl_offsets_s <= t_end_s)

                    imu_samples = {
                        "wall_ns": (
                            (video_wall_start + accl_offsets_s[mask]) * 1e9
                        ).astype(np.int64),
                        "accl": accl_xyz[mask],
                        "gyro": gyro_xyz[mask],
                    }

                video_sources.append(
                    {
                        "camera_idx": cam_id,
                        "video_path": str(video_path),
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "frame_wall_times_s": frame_wall_times_s,
                        "imu_samples": imu_samples,
                    }
                )

            all_episodes.append(
                {
                    "timestamps": episode_timestamps,
                    "robot_state": robot_state,
                    "video_sources": video_sources,
                }
            )

    if not all_episodes:
        raise click.ClickException("No valid episodes found.")

    logger.info(
        "Collected %d episodes, %d grippers, %d cameras.",
        len(all_episodes),
        n_grippers,
        n_cameras,
    )

    # Phase 2: Determine image dimensions and optional transforms
    first_video = all_episodes[0]["video_sources"][0]["video_path"]
    with av.open(first_video) as container:
        in_stream = container.streams.video[0]
        native_h, native_w = in_stream.height, in_stream.width
        video_fps = float(in_stream.average_rate)

    resize_tf = None
    if out_res_tuple is not None and fisheye_converter is None:
        resize_tf = get_image_transform(
            in_res=(native_w, native_h), out_res=out_res_tuple
        )
    image_w = out_res_tuple[0] if out_res_tuple is not None else native_w
    image_h = out_res_tuple[1] if out_res_tuple is not None else native_h

    is_mirror = None
    if mirror_swap:
        mirror_mask = np.ones((image_h, image_w, 3), dtype=np.uint8)
        mirror_mask = draw_predefined_mask(
            mirror_mask, color=(0, 0, 0), mirror=True, gripper=False, finger=False
        )
        is_mirror = mirror_mask[..., 0] == 0

    # Phase 3: Write one MCAP file per episode
    output_dir = output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = sum(len(ep["timestamps"]) for ep in all_episodes)
    logger.info(
        "Writing %d episode MCAP files (%d SLAM steps, %dx%d @ %.1f Hz, %s compression) to %s",
        len(all_episodes),
        total_steps,
        image_w,
        image_h,
        video_fps,
        compression,
        output_dir,
    )

    cfg = types.SimpleNamespace(
        inpaint_aruco=inpaint_aruco,
        mask_gripper=mask_gripper,
        no_mirror=no_mirror,
        mirror_swap=mirror_swap,
        jpeg_quality=jpeg_quality,
        slam_frame_stride=slam_frame_stride,
        fisheye_converter=fisheye_converter,
        resize_tf=resize_tf,
        is_mirror=is_mirror,
        output_dir=output_dir,
        compression_type=compression_type,
        compression=compression,
        n_grippers=n_grippers,
        n_cameras=n_cameras,
        image_w=image_w,
        image_h=image_h,
        video_fps=video_fps,
    )

    with tqdm(total=len(all_episodes), desc="Episodes") as pbar:
        # one episode per thread, each writes its own file
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for ep_idx, episode in enumerate(all_episodes):
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        f.result()  # re-raise any exception from write_episode
                    pbar.update(len(completed))

                futures.add(executor.submit(_write_episode, ep_idx, episode, cfg))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                f.result()  # re-raise any exception from write_episode
            pbar.update(len(completed))

    logger.info(
        "Done! %d episode MCAP files written to %s (%d total steps)",
        len(all_episodes),
        output_dir,
        total_steps,
    )


if __name__ == "__main__":
    main()
