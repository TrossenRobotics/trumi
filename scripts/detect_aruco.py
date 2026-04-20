"""
Detect and localize ArUco tags in a GoPro video, writing per-frame results to a pickle file.

:Steps:
    1. Load fisheye camera intrinsics and ArUco config.
    2. Iterate over video frames, applying a mirror mask to suppress false detections
       in reflections, and skipping frames according to slam_frame_stride.
    3. Detect and pose-estimate all visible ArUco tags per frame.
    4. Save per-frame results as a pickle file.

:Usage:
    uv run python scripts/detect_aruco.py -i <video.mp4> -o <output.pkl> \\
        -ci <intrinsics.json> -ac <aruco_config.yaml>
"""

import json
import logging
import pathlib
import pickle
import sys

import av
import click
import cv2
import numpy as np
import yaml
from tqdm import tqdm

from trumi.utils.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    draw_predefined_mask,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)

logger = logging.getLogger(__name__)


@click.command(help="Detect and localize ArUco tags in a GoPro video.")
@click.option("-i", "--input", required=True)
@click.option("-o", "--output", required=True)
@click.option("-ci", "--camera_intrinsics", required=True)
@click.option("-ac", "--aruco_yaml", required=True)
@click.option("-n", "--num_workers", type=int, default=4)
@click.option(
    "-sfs",
    "--slam_frame_stride",
    type=int,
    default=2,
    help="Process every Nth frame to match SLAM trajectory sampling rate "
    "(e.g. 2 for 120fps video where SLAM ran at 60fps).",
)
def main(input, output, camera_intrinsics, aruco_yaml, num_workers, slam_frame_stride):
    """Detect and localize ArUco tags in a GoPro video.

    :param input: Path to input video file.
    :param output: Path for output pickle file containing per-frame tag detections.
    :param camera_intrinsics: Path to fisheye camera intrinsics JSON file.
    :param aruco_yaml: Path to ArUco config YAML file.
    :param num_workers: Number of OpenCV threads for video decoding.
    :param slam_frame_stride: Process every Nth frame to align detection rate with
        SLAM trajectory sampling (e.g. 2 when SLAM subsampled from 120fps to 60fps).
    """
    cv2.setNumThreads(num_workers)

    # load aruco config
    aruco_config = parse_aruco_config(
        yaml.safe_load(pathlib.Path(aruco_yaml).read_text())
    )
    aruco_dict = aruco_config["aruco_dict"]
    marker_size_map = aruco_config["marker_size_map"]

    # load intrinsics
    raw_fisheye_intr = parse_fisheye_intrinsics(
        json.loads(pathlib.Path(camera_intrinsics).read_text())
    )

    results = list()
    with av.open(pathlib.Path(input).resolve()) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = np.array([in_stream.width, in_stream.height])
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res
        )

        n_frames = in_stream.frames
        logger.info("Processing %d frames (stride=%d)...", n_frames, slam_frame_stride)
        slam_idx = 0
        for i, frame in tqdm(
            enumerate(in_container.decode(in_stream)), total=n_frames, file=sys.stdout
        ):
            if i % slam_frame_stride != 0:
                continue
            img = frame.to_ndarray(format="bgr24")
            frame_cts_sec = frame.pts * in_stream.time_base
            # mask out mirrors to avoid false detections in reflections
            img = draw_predefined_mask(
                img, color=(0, 0, 0), mirror=True, gripper=False, finger=False
            )
            tag_dict = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True,
            )
            results.append(
                {
                    "frame_idx": slam_idx,
                    "time": float(frame_cts_sec),
                    "tag_dict": tag_dict,
                }
            )

            slam_idx += 1

    n_detections = sum(len(r["tag_dict"]) for r in results)
    logger.info(
        "Done: %d frames processed, %d tag detections total.",
        len(results),
        n_detections,
    )

    output_path = pathlib.Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pickle.dumps(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
