"""Generate a Zarr replay buffer from SLAM pipeline outputs.

:Steps:
    1. Load dataset_plan.pkl from each input directory and write low-dim
       robot state (eef pose, gripper width) into an in-memory Zarr
       ReplayBuffer.
    2. For every raw_video.mp4 referenced by the plan, decode the required
       frame ranges and, per frame:

       a. Inpaint detected ArUco tags.
       b. Mask out the gripper region.
       c. Resize the image to the output resolution.
       d. Optionally undistort via fisheye rectification.
       e. Optionally apply mirror-swap augmentation.
       f. Write the compressed image into the Zarr dataset.

    3. Save the completed ReplayBuffer as a .zarr.zip at --output.

:Usage:
    uv run python scripts/scripts_slam_pipeline/07_generate_replay_buffer.py \\
        -o <session_dir>/dataset.zarr.zip \\
        <session_dir>
"""

import concurrent.futures
import json
import logging
import multiprocessing
import pathlib
import pickle
from collections import defaultdict

import av
import click
import cv2
import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import JpegXl, register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer
from trumi.utils.cv_util import (
    FisheyeRectConverter,
    draw_predefined_mask,
    get_image_transform,
    inpaint_tag,
    parse_fisheye_intrinsics,
)

register_codecs()

logger = logging.getLogger(__name__)


@click.command(help="Generate a Zarr replay buffer from SLAM pipeline outputs.")
@click.argument("input", nargs=-1)
@click.option("-o", "--output", required=True, help="Output .zarr.zip path.")
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
    "-cl",
    "--compression_level",
    type=int,
    default=99,
    show_default=True,
    help="JpegXl compression level.",
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
    "-n",
    "--num_workers",
    type=int,
    default=None,
    help="Number of parallel video-decoding threads. Defaults to CPU count.",
)
@click.option(
    "-ci",
    "--camera_intrinsics",
    default=None,
    help="Path to fisheye intrinsics JSON. Required when --out_fov is set.",
)
@click.option(
    "-sfs",
    "--slam_frame_stride",
    type=int,
    default=2,
    help="Frame stride used by SLAM and ArUco detection (e.g. 2 when raw video is 120fps and SLAM ran at 60fps).",
)
def main(
    input,
    output,
    out_res,
    out_fov,
    camera_intrinsics,
    compression_level,
    no_mirror,
    mirror_swap,
    num_workers,
    slam_frame_stride,
):
    """Generate a Zarr replay buffer from SLAM pipeline outputs.

    :param input: One or more input directories containing dataset_plan.pkl.
    :param output: Output .zarr.zip path.
    :param out_res: Output image resolution as 'W,H' string.
    :param out_fov: Vertical FOV for fisheye rectification (degrees). If omitted
        images are resized without undistortion.
    :param camera_intrinsics: Path to fisheye intrinsics JSON. Required when out_fov is set.
    :param compression_level: JpegXl compression level.
    :param no_mirror: Mask out mirror regions from observations.
    :param mirror_swap: Apply mirror-swap augmentation.
    :param num_workers: Number of parallel video-decoding threads.
    :param slam_frame_stride: Frame stride matching SLAM and ArUco detection rate.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if pathlib.Path(output).is_file():
        if click.confirm(f"Output file {output} exists! Overwrite?", abort=True):
            pass

    out_res = tuple(int(x) for x in out_res.split(","))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
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

    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

    # dump lowdim data to replay buffer
    # generate argument for videos
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    for ipath in input:
        ipath = pathlib.Path(ipath).expanduser().resolve()
        demos_path = ipath.joinpath("demos")
        plan_path = ipath.joinpath("dataset_plan.pkl")
        if not plan_path.is_file():
            logger.info("Skipping %s: no dataset_plan.pkl", ipath.name)
            continue

        plan = pickle.loads(plan_path.read_bytes())

        videos_dict = defaultdict(list)
        for plan_episode in plan:
            grippers = plan_episode["grippers"]

            # check that all episodes have the same number of grippers
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)

            cameras = plan_episode["cameras"]
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)

            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):
                eef_pose = gripper["tcp_pose"]
                eef_pos = eef_pose[..., :3]
                eef_rot = eef_pose[..., 3:]
                gripper_widths = gripper["gripper_width"]
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper["demo_start_pose"]
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper["demo_end_pose"]

                robot_name = f"robot{gripper_id}"
                episode_data[robot_name + "_eef_pos"] = eef_pos.astype(np.float32)
                episode_data[robot_name + "_eef_rot_axis_angle"] = eef_rot.astype(
                    np.float32
                )
                episode_data[robot_name + "_gripper_width"] = np.expand_dims(
                    gripper_widths, axis=-1
                ).astype(np.float32)
                episode_data[robot_name + "_demo_start_pose"] = demo_start_pose
                episode_data[robot_name + "_demo_end_pose"] = demo_end_pose

            out_replay_buffer.add_episode(data=episode_data, compressors=None)

            # aggregate video gen arguments
            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera["video_path"]
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()

                video_start, video_end = camera["video_start_end"]
                if n_frames is None:
                    # video_start_end are raw-fps frame indices; divide by stride
                    # to get slam-fps frame count matching the lowdim data
                    n_frames = (video_end - video_start) // slam_frame_stride
                else:
                    assert n_frames == (video_end - video_start) // slam_frame_stride

                videos_dict[str(video_path)].append(
                    {
                        "camera_idx": cam_id,
                        "frame_start": video_start,
                        "frame_end": video_end,
                        "buffer_start": buffer_start,
                    }
                )
            buffer_start += n_frames

        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())

    logger.info("%d videos used in total!", len(all_videos))

    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width

    # dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f"camera{cam_id}_rgb"
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            # (total_frames, out_res, 3)
            shape=(out_replay_buffer["robot0_eef_pos"].shape[0],) + out_res + (3,),
            # (1, out_res, 3)
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8,
        )

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        pkl_path = pathlib.Path(mp4_path).parent / "tag_detection.pkl"
        tag_detection_results = pickle.loads(pkl_path.read_bytes())
        resize_tf = get_image_transform(in_res=(iw, ih), out_res=out_res)
        tasks = sorted(tasks, key=lambda x: x["frame_start"])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task["camera_idx"]
            else:
                assert camera_idx == task["camera_idx"]
        name = f"camera{camera_idx}_rgb"
        img_array = replay_buffer.data[name]

        curr_task_idx = 0

        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh, ow, 3), dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0, 0, 0), mirror=True, gripper=False, finger=False
            )
            is_mirror = mirror_mask[..., 0] == 0

        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(
                enumerate(container.decode(in_stream)),
                total=in_stream.frames,
                leave=False,
            ):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break

                if frame_idx < tasks[curr_task_idx]["frame_start"]:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]["frame_end"]:
                    if frame_idx == tasks[curr_task_idx]["frame_start"]:
                        buffer_idx = tasks[curr_task_idx]["buffer_start"]

                    # only process every Nth raw frame to match slam-fps lowdim data
                    if (
                        frame_idx - tasks[curr_task_idx]["frame_start"]
                    ) % slam_frame_stride == 0:
                        img = frame.to_ndarray(format="rgb24")

                        # inpaint tags
                        this_det = tag_detection_results[frame_idx // slam_frame_stride]
                        all_corners = [
                            x["corners"] for x in this_det["tag_dict"].values()
                        ]
                        for corners in all_corners:
                            img = inpaint_tag(img, corners)

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

                        # compress image
                        img_array[buffer_idx] = img
                        buffer_idx += 1

                    if (frame_idx + 1) == tasks[curr_task_idx]["frame_end"]:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    raise RuntimeError(
                        f"Unexpected frame_idx {frame_idx} outside all task ranges"
                    )

    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    pbar.update(len(completed))

                futures.add(
                    executor.submit(video_to_zarr, out_replay_buffer, mp4_path, tasks)
                )

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    # dump to disk
    logger.info("Saving ReplayBuffer to %s", output)
    with zarr.ZipStore(output, mode="w") as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    logger.info("Done! %d videos used in total!", len(all_videos))


if __name__ == "__main__":
    main()
