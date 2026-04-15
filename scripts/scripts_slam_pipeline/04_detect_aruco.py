"""
Run ArUco tag detection in batch over all demo videos in a demos directory.

:Steps:
    1. Discover all */raw_video.mp4 paths under input_dir.
    2. Validate camera intrinsics and ArUco config files exist.
    3. Submit per-video detection jobs to a thread pool.
    4. Output per-video tag_detection.pkl files.

:Usage:
    uv run python scripts/scripts_slam_pipeline/04_detect_aruco.py --input_dir <session_dir/demos/> \
        --camera_intrinsics <intrinsics.json> --aruco_yaml <config.yaml>
"""

import concurrent.futures
import logging
import multiprocessing
import pathlib
import subprocess
import sys

import click
from tqdm import tqdm

logger = logging.getLogger(__name__)


def runner(cmd, stdout_path, stderr_path):
    """Run a detect_aruco subprocess, capturing stdout and stderr.

    :param cmd: Command list passed to subprocess.run.
    :param stdout_path: File path to write stdout to.
    :param stderr_path: File path to write stderr to.
    :return: subprocess.CompletedProcess result.
    """
    with stdout_path.open("w") as stdout_f, stderr_path.open("w") as stderr_f:
        result = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f)
    if result.returncode != 0:
        logger.error(
            "detect_aruco failed for %s, see %s", stdout_path.parent.name, stderr_path
        )
    return result


@click.command(
    help="Run ArUco tag detection over all demo videos in a session directory."
)
@click.option(
    "-i",
    "--input_dir",
    required=True,
    help="Demos directory (session_dir/demos/) containing */raw_video.mp4 paths",
)
@click.option(
    "-ci",
    "--camera_intrinsics",
    required=True,
    help="Fisheye camera intrinsics JSON (2.7k)",
)
@click.option("-ac", "--aruco_yaml", required=True, help="ArUco config YAML file")
@click.option(
    "-n",
    "--num_workers",
    type=int,
    default=None,
    help="Parallel workers. Defaults to cpu_count // 2.",
)
@click.option(
    "-sfs",
    "--slam_frame_stride",
    type=int,
    default=2,
    help="Process every Nth frame to match SLAM trajectory sampling rate "
    "(e.g. 2 for 120fps video where SLAM ran at 60fps).",
)
def main(input_dir, camera_intrinsics, aruco_yaml, num_workers, slam_frame_stride):
    """Run ArUco tag detection in batch over all demo videos under input_dir.

    :param input_dir: Demos directory (session_dir/demos/) containing */raw_video.mp4 paths.
    :param camera_intrinsics: Path to fisheye camera intrinsics JSON file (2.7k resolution).
    :param aruco_yaml: Path to ArUco config YAML file.
    :param num_workers: Parallel workers. Defaults to cpu_count // 2.
    :param slam_frame_stride: Process every Nth frame to align detection rate with
        SLAM trajectory sampling (e.g. 2 when SLAM subsampled 120fps to 60fps).
    """
    input_dir = pathlib.Path(input_dir).resolve()
    input_video_dirs = [x.parent for x in input_dir.glob("*/raw_video.mp4")]
    logger.info("Found %d video dirs", len(input_video_dirs))

    camera_intrinsics = pathlib.Path(camera_intrinsics).resolve()
    aruco_yaml_path = pathlib.Path(aruco_yaml).resolve()
    if not camera_intrinsics.is_file():
        raise click.ClickException(f"Camera intrinsics not found: {camera_intrinsics}")
    if not aruco_yaml_path.is_file():
        raise click.ClickException(f"ArUco config not found: {aruco_yaml_path}")

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    script_path = pathlib.Path(__file__).parent.parent.joinpath("detect_aruco.py")

    with tqdm(total=len(input_video_dirs)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            all_futures = set()
            for video_dir in input_video_dirs:
                video_dir = video_dir.resolve()
                pkl_path = video_dir.joinpath("tag_detection.pkl")
                if pkl_path.is_file():
                    logger.info(
                        "tag_detection.pkl already exists, skipping %s", video_dir.name
                    )
                    pbar.update(1)
                    continue

                video_path = video_dir.joinpath("raw_video.mp4")
                stdout_path = video_dir.joinpath("detect_aruco_stdout.txt")
                stderr_path = video_dir.joinpath("detect_aruco_stderr.txt")

                cmd = [
                    sys.executable,
                    str(script_path),
                    "--input",
                    str(video_path),
                    "--output",
                    str(pkl_path),
                    "--camera_intrinsics",
                    str(camera_intrinsics),
                    "--aruco_yaml",
                    str(aruco_yaml_path),
                    "--num_workers",
                    "1",
                    "--slam_frame_stride",
                    str(slam_frame_stride),
                ]

                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    pbar.update(len(completed))

                fut = executor.submit(runner, cmd, stdout_path, stderr_path)
                futures.add(fut)
                all_futures.add(fut)

            if futures:
                completed, _ = concurrent.futures.wait(futures)
                pbar.update(len(completed))

    results = [f.result() for f in all_futures]
    n_ok = sum(
        1
        for r in results
        if isinstance(r, subprocess.CompletedProcess) and r.returncode == 0
    )
    n_fail = len(results) - n_ok
    logger.info(
        "Done: %d succeeded, %d failed (logs: detect_aruco_stdout.txt / detect_aruco_stderr.txt)",
        n_ok,
        n_fail,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
