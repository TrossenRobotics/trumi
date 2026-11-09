"""Generate a LeRobot v3 dataset from SLAM pipeline outputs.

Requires the lerobot extra: ``uv sync --extra lerobot``.

:Steps:
    1. Load dataset_plan.pkl from each input directory.
    2. For every episode, decode the required frame range of each camera and,
       per frame:

       a. Inpaint detected ArUco tags.
       b. Mask out the gripper region.
       c. Resize the image (--out_res).
       d. Optionally undistort via fisheye rectification.
       e. Optionally apply mirror-swap augmentation.

    3. Write one LeRobot frame per SLAM step with the camera images, the
       action (eef pose and gripper width) and the demo start/end poses.

:Features:
    observation.images.camera{N}  — video, (H, W, 3)
    action                        — float32, per gripper [x, y, z, ax, ay, az, width]
    robot{N}_demo_start_pose      — float32, (6,) [x, y, z, ax, ay, az]
    robot{N}_demo_end_pose        — float32, (6,) [x, y, z, ax, ay, az]

:Usage:
    uv run python scripts/scripts_slam_pipeline/07_generate_lerobot_dataset.py \\
        -o <session_dir>/dataset_lerobot \\
        <session_dir>
"""

import json
import logging
import pathlib
import pickle
import shutil

import av
import click
import cv2
import numpy as np
from lerobot.configs import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from tqdm import tqdm

from trumi.utils.cv_util import (
    FisheyeRectConverter,
    draw_predefined_mask,
    get_image_transform,
    inpaint_tag,
    parse_fisheye_intrinsics,
)

logger = logging.getLogger(__name__)

ACTION_NAMES = [
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_rot_axis_angle_x",
    "eef_rot_axis_angle_y",
    "eef_rot_axis_angle_z",
    "gripper_width",
]
POSE_NAMES = ["x", "y", "z", "ax", "ay", "az"]


@click.command(help="Generate a LeRobot v3 dataset from SLAM pipeline outputs.")
@click.argument("input_dirs", nargs=-1)
@click.option(
    "-o", "--output", required=True, help="Output directory for the LeRobot dataset."
)
@click.option(
    "-r",
    "--repo_id",
    default=None,
    help="Dataset repo id, e.g. <user>/<dataset>. Defaults to trumi/<output dir name>.",
)
@click.option(
    "-t",
    "--task",
    default="",
    help="Natural-language task description stored with every frame.",
)
@click.option(
    "-or",
    "--out_res",
    type=str,
    default="224,224",
    show_default=True,
    help="Output image resolution as W,H.",
)
@click.option(
    "-of",
    "--out_fov",
    type=float,
    default=None,
    help="Vertical FOV (degrees) for fisheye rectification. Omit to resize only.",
)
@click.option(
    "-ci",
    "--camera_intrinsics",
    default=None,
    help="Path to fisheye intrinsics JSON. Required when --out_fov is set.",
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
    "-sfs",
    "--slam_frame_stride",
    type=int,
    default=2,
    help="Frame stride used by SLAM and ArUco detection (e.g. 2 when raw video is 120fps and SLAM ran at 60fps).",
)
def main(
    input_dirs,
    output,
    repo_id,
    task,
    out_res,
    out_fov,
    camera_intrinsics,
    no_mirror,
    mirror_swap,
    slam_frame_stride,
):
    """Generate a LeRobot v3 dataset from SLAM pipeline outputs.

    :param input_dirs: One or more input directories containing dataset_plan.pkl.
    :param output: Output directory for the LeRobot dataset.
    :param repo_id: Dataset repo id. Defaults to trumi/<output dir name>.
    :param task: Natural-language task description stored with every frame.
    :param out_res: Output image resolution as 'W,H' string.
    :param out_fov: Vertical FOV for fisheye rectification (degrees). If omitted
        images are resized without undistortion.
    :param camera_intrinsics: Path to fisheye intrinsics JSON. Required when out_fov is set.
    :param no_mirror: Mask out mirror regions from observations.
    :param mirror_swap: Apply mirror-swap augmentation.
    :param slam_frame_stride: Frame stride matching SLAM and ArUco detection rate.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", force=True
    )

    output = pathlib.Path(output).expanduser().resolve()
    if output.exists():
        if click.confirm(f"Output directory {output} exists! Overwrite?", abort=True):
            shutil.rmtree(output)
    if repo_id is None:
        repo_id = f"trumi/{output.name}"

    out_res = tuple(int(x) for x in out_res.split(","))
    cv2.setNumThreads(1)

    fisheye_converter = None
    # process raw undistorted fisheye image if out_fov is None
    if out_fov is not None:
        if camera_intrinsics is None:
            raise click.UsageError(
                "--camera_intrinsics is required when --out_fov is set."
            )
        intr_path = pathlib.Path(camera_intrinsics).expanduser().resolve()
        if not intr_path.is_file():
            raise click.ClickException(f"Camera intrinsics not found: {intr_path}")
        opencv_intr_dict = parse_fisheye_intrinsics(json.loads(intr_path.read_text()))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict, out_size=out_res, out_fov=out_fov
        )

    # collect episodes from all plans
    episodes = list()
    for ipath in input_dirs:
        ipath = pathlib.Path(ipath).expanduser().resolve()
        plan_path = ipath.joinpath("dataset_plan.pkl")
        if not plan_path.is_file():
            logger.info("Skipping %s: no dataset_plan.pkl", ipath.name)
            continue
        plan = pickle.loads(plan_path.read_bytes())
        episodes.extend((ipath.joinpath("demos"), ep) for ep in plan)

    if not episodes:
        raise click.ClickException(
            "No valid episodes found. Check that input directories contain dataset_plan.pkl."
        )

    # check that all episodes have the same number of grippers and cameras
    n_grippers = len(episodes[0][1]["grippers"])
    n_cameras = len(episodes[0][1]["cameras"])
    for _, plan_episode in episodes:
        if len(plan_episode["grippers"]) != n_grippers:
            raise click.ClickException(
                f"Inconsistent gripper count: expected {n_grippers}, got {len(plan_episode['grippers'])}."
            )
        if len(plan_episode["cameras"]) != n_cameras:
            raise click.ClickException(
                f"Inconsistent camera count: expected {n_cameras}, got {len(plan_episode['cameras'])}."
            )

    # lerobot needs an integer fps; use the SLAM rate of the plan
    episode_timestamps = episodes[0][1]["episode_timestamps"]
    fps = int(round(1.0 / np.median(np.diff(episode_timestamps))))
    logger.info(
        "Found %d episodes, %d grippers, %d cameras at %d fps.",
        len(episodes),
        n_grippers,
        n_cameras,
        fps,
    )

    # out_res is (W, H); images are stored in NumPy order (H, W, C)
    out_img_shape = (out_res[1], out_res[0], 3)
    features = dict()
    for cam_id in range(n_cameras):
        features[f"observation.images.camera{cam_id}"] = {
            "dtype": "video",
            "shape": out_img_shape,
            "names": ["height", "width", "channels"],
        }
    features["action"] = {
        "dtype": "float32",
        "shape": (len(ACTION_NAMES) * n_grippers,),
        "names": [
            f"robot{gripper_id}_{name}"
            for gripper_id in range(n_grippers)
            for name in ACTION_NAMES
        ],
    }
    for gripper_id in range(n_grippers):
        for key in ("demo_start_pose", "demo_end_pose"):
            features[f"robot{gripper_id}_{key}"] = {
                "dtype": "float32",
                "shape": (len(POSE_NAMES),),
                "names": POSE_NAMES,
            }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output,
        robot_type="trumi",
        use_videos=True,
        # h264 so the LeRobot dataset visualizer can play the videos
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        # write frames in background threads; videos are encoded per episode.
        # streaming_encoding is not used because it drops frames when the
        # encoder falls behind.
        image_writer_threads=4,
    )

    is_mirror = None
    if mirror_swap:
        ow, oh = out_res
        mirror_mask = np.ones((oh, ow, 3), dtype=np.uint8)
        mirror_mask = draw_predefined_mask(
            mirror_mask, color=(0, 0, 0), mirror=True, gripper=False, finger=False
        )
        is_mirror = mirror_mask[..., 0] == 0

    def iter_camera_frames(video_path, video_start, video_end):
        """Yield every slam_frame_stride-th processed frame in [video_start, video_end)."""
        pkl_path = video_path.parent / "tag_detection.pkl"
        if not pkl_path.is_file():
            raise click.ClickException(
                f"tag_detection.pkl not found for video {video_path}. "
                f"Expected at: {pkl_path}"
            )
        tag_detection_results = pickle.loads(pkl_path.read_bytes())

        with av.open(str(video_path)) as container:
            in_stream = container.streams.video[0]
            in_stream.thread_type = "AUTO"
            resize_tf = get_image_transform(
                in_res=(in_stream.width, in_stream.height), out_res=out_res
            )
            for frame_idx, frame in enumerate(container.decode(in_stream)):
                if frame_idx >= video_end:
                    break
                if frame_idx < video_start:
                    continue
                # only process every Nth raw frame to match slam-fps lowdim data
                if (frame_idx - video_start) % slam_frame_stride != 0:
                    continue
                img = frame.to_ndarray(format="rgb24")

                # inpaint tags
                this_det = tag_detection_results[frame_idx // slam_frame_stride]
                for tag in this_det["tag_dict"].values():
                    img = inpaint_tag(img, tag["corners"])

                # mask out gripper
                img = draw_predefined_mask(
                    img,
                    color=(0, 0, 0),
                    mirror=no_mirror,
                    gripper=True,
                    finger=False,
                )
                # resize
                if fisheye_converter is None:
                    img = resize_tf(img)
                else:
                    img = fisheye_converter.forward(img)

                # handle mirror swap
                if mirror_swap:
                    img[is_mirror] = img[:, ::-1, :][is_mirror]

                yield img

    for demos_path, plan_episode in tqdm(episodes, desc="Episodes"):
        grippers = plan_episode["grippers"]
        n_frames = len(plan_episode["episode_timestamps"])

        actions = np.concatenate(
            [
                np.concatenate(
                    [
                        gripper["tcp_pose"],
                        np.expand_dims(gripper["gripper_width"], axis=-1),
                    ],
                    axis=-1,
                )
                for gripper in grippers
            ],
            axis=-1,
        ).astype(np.float32)

        camera_frames = list()
        for camera in plan_episode["cameras"]:
            video_path = demos_path.joinpath(camera["video_path"]).absolute()
            if not video_path.is_file():
                raise click.ClickException(f"Video file not found: {video_path}")
            video_start, video_end = camera["video_start_end"]
            camera_frames.append(iter_camera_frames(video_path, video_start, video_end))

        n_added = 0
        for i, images in zip(range(n_frames), zip(*camera_frames)):
            frame = {"task": task, "action": actions[i]}
            for cam_id, img in enumerate(images):
                frame[f"observation.images.camera{cam_id}"] = img
            for gripper_id, gripper in enumerate(grippers):
                robot_name = f"robot{gripper_id}"
                frame[robot_name + "_demo_start_pose"] = gripper[
                    "demo_start_pose"
                ].astype(np.float32)
                frame[robot_name + "_demo_end_pose"] = gripper["demo_end_pose"].astype(
                    np.float32
                )
            dataset.add_frame(frame)
            n_added += 1
        if n_added < n_frames:
            raise click.ClickException(
                f"Frame count mismatch in {demos_path}: "
                f"expected {n_frames}, got {n_added}."
            )
        dataset.save_episode()

    # closes the parquet writers; required before the dataset can be loaded
    dataset.finalize()
    logger.info("Done! %d episodes written to %s", len(episodes), output)


if __name__ == "__main__":
    main()
