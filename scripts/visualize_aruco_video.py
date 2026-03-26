"""Render ArUco tag detections overlaid on a video and write the result to a new file.

:Steps:
    1. Load tag_detection.pkl (must already exist alongside the video).
    2. Decode the source video frame by frame.
    3. For each frame with detections: draw tag corners, IDs, and 3D pose axes.
    4. Write the annotated frames to an output MP4.

:Usage:
    uv run python scripts/visualize_aruco_video.py \\
        -i path/to/demo_dir \\
        -ci path/to/intrinsics.json \\
        [-sfs 2] \\
        -o path/to/output.mp4
"""

import json
import pathlib
import pickle
import sys

import av
import click
import cv2
import numpy as np
from tqdm import tqdm

from utils.common.cv_util import (
    convert_fisheye_intrinsics_resolution,
    parse_fisheye_intrinsics,
)

# Length of the 3D pose axis arrows drawn on each tag, in metres.
AXIS_LENGTH_M = 0.03


def _build_frame_lookup(detections: list) -> dict:
    """Build a dict mapping frame_idx to tag_dict."""
    return {entry["frame_idx"]: entry["tag_dict"] for entry in detections}


def _draw_detections(
    img_bgr: np.ndarray, tag_dict: dict, K: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """Draw corners, IDs and pose axes for all detected tags onto img_bgr in-place.

    :param img_bgr: BGR image to annotate.
    :param tag_dict: Mapping of tag_id to {corners, rvec, tvec}.
    :param K: 3*3 fisheye camera matrix.
    :param D: Fisheye distortion coefficients (4*1).
    :return: Annotated BGR image.
    """
    corners_list = []
    ids_list = []
    for tid, info in tag_dict.items():
        corners_list.append(info["corners"].reshape(1, 4, 2).astype(np.float32))
        ids_list.append([tid])

    if corners_list:
        cv2.aruco.drawDetectedMarkers(img_bgr, corners_list, np.array(ids_list))

    # 3D axis endpoints in marker frame: X, Y, Z tips + origin.
    # rvec/tvec came from solveOnP on undistorted pinhole coords, so we must
    # project back into the fisheye image using cv2.fisheye.projectPoints.
    axis_3d = np.float32(
        [[AXIS_LENGTH_M, 0, 0], [0, AXIS_LENGTH_M, 0], [0, 0, AXIS_LENGTH_M], [0, 0, 0]]
    ).reshape(-1, 1, 3)
    for info in tag_dict.values():
        rvec = info["rvec"].reshape(3, 1)
        tvec = info["tvec"].reshape(3, 1)
        pts, _ = cv2.fisheye.projectPoints(axis_3d, rvec, tvec, K, D)
        pts = pts.reshape(-1, 2).astype(int)
        origin = tuple(pts[3])
        cv2.arrowedLine(
            img_bgr, origin, tuple(pts[0]), (0, 0, 255), 2, tipLength=0.2
        )  # X red
        cv2.arrowedLine(
            img_bgr, origin, tuple(pts[1]), (0, 255, 0), 2, tipLength=0.2
        )  # Y green
        cv2.arrowedLine(
            img_bgr, origin, tuple(pts[2]), (255, 0, 0), 2, tipLength=0.2
        )  # Z blue

    return img_bgr


@click.command()
@click.option(
    "-i",
    "--input_dir",
    required=True,
    type=click.Path(exists=True),
    help="Demo directory containing raw_video.mp4 and tag_detection.pkl.",
)
@click.option(
    "-ci",
    "--camera_intrinsics",
    required=True,
    type=click.Path(exists=True),
    help="Fisheye camera intrinsics JSON (2.7k).",
)
@click.option(
    "-o",
    "--output_video",
    default=None,
    help="Output MP4 path. Defaults to <input_dir>/aruco_overlay.mp4.",
)
@click.option(
    "-sfs",
    "--slam_frame_stride",
    type=int,
    default=2,
    help="Only include every Nth frame, matching the stride used during detection.",
)
def main(input_dir, camera_intrinsics, output_video, slam_frame_stride):
    """Render ArUco detections overlaid on a video.

    :param input_dir: Directory containing raw_video.mp4 and tag_detection.pkl.
    :param camera_intrinsics: Path to fisheye camera intrinsics JSON.
    :param output_video: Output MP4 path.
    :param slam_frame_stride: Only write every Nth frame to the output video,
        matching the stride used when running detect_aruco.
    """
    input_dir = pathlib.Path(input_dir).resolve()
    input_video = input_dir / "raw_video.mp4"
    pkl_path = input_dir / "tag_detection.pkl"

    if not input_video.is_file():
        raise click.ClickException(f"raw_video.mp4 not found in {input_dir}")
    if not pkl_path.is_file():
        raise click.ClickException(f"tag_detection.pkl not found in {input_dir}")

    if output_video is None:
        output_video = input_dir / "aruco_overlay.mp4"
    output_video = pathlib.Path(output_video)

    with open(pkl_path, "rb") as f:
        detections = pickle.load(f)
    frame_lookup = _build_frame_lookup(detections)

    raw_intr = parse_fisheye_intrinsics(
        json.loads(pathlib.Path(camera_intrinsics).read_text())
    )
    # D does not change with resolution (Kannala-Brandt params are resolution-independent).
    D = raw_intr["D"]

    n_drawn = 0
    with av.open(str(input_video)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        n_frames = in_stream.frames

        in_res = np.array([in_stream.height, in_stream.width])[::-1]
        K = convert_fisheye_intrinsics_resolution(raw_intr, in_res)["K"]

        with av.open(str(output_video), mode="w") as out_container:
            out_stream = out_container.add_stream("h264", rate=in_stream.average_rate)
            out_stream.width = in_stream.width
            out_stream.height = in_stream.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "18"}

            print(
                f"Writing {n_frames // slam_frame_stride} frames to {output_video.name} (every {slam_frame_stride} of {n_frames} raw frames)"
            )
            for frame_idx, frame in tqdm(
                enumerate(in_container.decode(in_stream)),
                total=n_frames,
                file=sys.stdout,
            ):
                if frame_idx % slam_frame_stride != 0:
                    continue
                img = frame.to_ndarray(format="bgr24")

                tag_dict = frame_lookup.get(frame_idx // slam_frame_stride)
                if tag_dict:
                    img = _draw_detections(img, tag_dict, K, D)
                    n_drawn += len(tag_dict)

                out_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                out_frame.pts = frame.pts
                out_frame.time_base = frame.time_base
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)

            for packet in out_stream.encode():
                out_container.mux(packet)

    # Per-tag detection coverage stats.
    TAG_LABELS = {0: "gripper left", 1: "gripper right"}
    report_ids = [
        tid
        for tid in sorted({tid for e in detections for tid in e["tag_dict"]})
        if tid in TAG_LABELS
    ]
    print(f"\nDetection coverage ({len(detections)} frames total):")
    for tid in report_ids:
        detected = sum(1 for e in detections if tid in e["tag_dict"])
        missing = len(detections) - detected
        print(
            f"  tag {tid:>3d} ({TAG_LABELS[tid]:<14}): {detected} detected, {missing} missing ({100 * missing / len(detections):.1f}% missing)"
        )
    print(f"\nDone: {n_drawn} tag detections drawn on {output_video}")


if __name__ == "__main__":
    main()
