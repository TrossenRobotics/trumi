"""
Run ORB-SLAM3 localization on all demo videos in a demos directory using a
pre-built map atlas and generate camera_trajectory.csv file.

:Steps:
    1. Find all demo videos in the demos dir. Verify the pre-built map atlas exists.
    2. For each video, generate a SLAM mask (mirror + finger regions), then submit a
       Docker SLAM job to a thread pool. The script automatically subsamples frames so
       IMU samples per frame >= 3. (eg. every 2nd frame at 120 fps so running SLAM at 60 fps.)
    3. Output per-video camera_trajectory.csv files.

:Usage:
    uv run python scripts/scripts_slam_pipeline/03_batch_slam.py --input_dir <demos_dir> [--map_path <map.osa>]
"""

import concurrent.futures
import logging
import multiprocessing
import pathlib
import subprocess

import av
import click
import cv2
import numpy as np
from tqdm import tqdm

from trumi.utils.cv_util import draw_predefined_mask

logger = logging.getLogger(__name__)

# GoPro 2.7k resolution used for per-video SLAM masks
GOPRO_2_7K_H, GOPRO_2_7K_W = 2028, 2704


def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
    """Run a subprocess command capturing stdout/stderr, returning on timeout instead of raising.

    :param cmd: Command list passed to subprocess.run.
    :param cwd: Working directory for the subprocess.
    :param stdout_path: File path to write stdout to.
    :param stderr_path: File path to write stderr to.
    :param timeout: Maximum seconds to wait before killing the process.
    :return: subprocess.CompletedProcess on success, subprocess.TimeoutExpired on timeout.
    """
    try:
        with stdout_path.open("w") as stdout_f, stderr_path.open("w") as stderr_f:
            return subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=stdout_f,
                stderr=stderr_f,
                timeout=timeout,
                **kwargs,
            )
    except subprocess.TimeoutExpired as e:
        return e


@click.command(
    help="Run ORB-SLAM3 localization in batch over all demo videos to generate camera_trajectory.csv files."
)
@click.option(
    "-i",
    "--input_dir",
    required=True,
    help="Demos directory containing demo*/raw_video.mp4 paths",
)
@click.option(
    "-m",
    "--map_path",
    default=None,
    help="ORB_SLAM3 *.osa map atlas for localization. Defaults to <input_dir>/mapping/map_atlas.osa",
)
@click.option("-d", "--docker_image", default="orb_slam3")
@click.option(
    "-n",
    "--num_workers",
    type=click.IntRange(min=1),
    default=None,
    help="Parallel Docker workers. Defaults to max(1, cpu_count // 2).",
)
@click.option(
    "-ml",
    "--max_lost_frames",
    type=int,
    default=60,
    help="Terminate SLAM if tracking is lost for this many consecutive frames.",
)
@click.option(
    "-tm",
    "--timeout_multiple",
    type=float,
    default=16,
    help="Per-video timeout = video_duration_s * timeout_multiple. Increase for slow machines.",
)
@click.option(
    "-nm",
    "--no_mask",
    is_flag=True,
    default=False,
    help="Whether to mask out finger and mirrors. Set if map is created with bare GoPro not on gripper.",
)
def main(
    input_dir,
    map_path,
    docker_image,
    num_workers,
    max_lost_frames,
    timeout_multiple,
    no_mask,
):
    """Run ORB-SLAM3 localization in batch over all demo videos under input_dir to generate camera_trajectory.csv files.

    Discovers raw_video.mp4 paths, generates per-video SLAM masks, and submits
    Docker SLAM jobs in parallel using a thread pool.

    :param input_dir: Demos directory containing raw_video.mp4 paths.
    :param map_path: Path to the ORB-SLAM3 map atlas (*.osa) for localization. Defaults to
        <input_dir>/mapping/map_atlas.osa.
    :param docker_image: Name of the Docker image containing the ORB-SLAM3 binary.
    :param num_workers: Parallel Docker workers. Defaults to cpu_count // 2.
    :param max_lost_frames: Terminate SLAM if tracking is lost for this many consecutive frames.
    :param timeout_multiple: Per-video timeout = video_duration_s * timeout_multiple.
    :param no_mask: If True, skip mask generation and run SLAM on the raw video.
    """
    input_dir = pathlib.Path(input_dir).resolve()
    input_video_dirs = [x.parent for x in input_dir.glob("demo*/raw_video.mp4")]
    input_video_dirs += [x.parent for x in input_dir.glob("map*/raw_video.mp4")]
    logger.info("Found %d video dirs", len(input_video_dirs))

    if map_path is None:
        map_path = input_dir.joinpath("mapping", "map_atlas.osa")
    else:
        map_path = pathlib.Path(map_path).resolve()
    if not map_path.is_file():
        raise click.ClickException(f"Map atlas not found: {map_path}")

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    with tqdm(total=len(input_video_dirs)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            all_futures = set()
            for video_dir in input_video_dirs:
                video_dir = video_dir.resolve()
                if video_dir.joinpath("camera_trajectory.csv").is_file():
                    logger.info(
                        "camera_trajectory.csv already exists, skipping %s",
                        video_dir.name,
                    )
                    pbar.update(1)
                    continue

                # Validate imu_data.json exists before submitting job
                imu_json_path = video_dir.joinpath("imu_data.json")
                if not imu_json_path.is_file():
                    logger.warning("imu_data.json missing, skipping %s", video_dir.name)
                    pbar.update(1)
                    continue

                # softlink won't work in bind volume
                mount_target = pathlib.Path("/data")
                csv_path = mount_target.joinpath("camera_trajectory.csv")
                video_path = mount_target.joinpath("raw_video.mp4")
                json_path = mount_target.joinpath("imu_data.json")
                mask_path = mount_target.joinpath("slam_mask.png")
                mask_write_path = video_dir.joinpath("slam_mask.png")

                # find video duration
                with av.open(
                    str(video_dir.joinpath("raw_video.mp4").resolve())
                ) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple

                if not no_mask:
                    slam_mask = np.zeros((GOPRO_2_7K_H, GOPRO_2_7K_W), dtype=np.uint8)
                    slam_mask = draw_predefined_mask(
                        slam_mask, color=255, mirror=True, gripper=False, finger=True
                    )
                    cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                map_mount_source = map_path
                map_mount_target = pathlib.Path("/map").joinpath(map_mount_source.name)

                # run SLAM
                cmd = [
                    "docker",
                    "run",
                    "--rm",  # delete after finish
                    "--volume",
                    str(video_dir) + ":" + "/data",
                    "--volume",
                    str(map_mount_source.parent) + ":" + str(map_mount_target.parent),
                    docker_image,
                    "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
                    "--vocabulary",
                    "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
                    "--setting",
                    "/ORB_SLAM3/Examples/Monocular-Inertial/gopro13_maxlens_fisheye_setting_v1_720.yaml",
                    "--input_video",
                    str(video_path),
                    "--input_imu_json",
                    str(json_path),
                    "--output_trajectory_csv",
                    str(csv_path),
                    "--load_map",
                    str(map_mount_target),
                    "--max_lost_frames",
                    str(max_lost_frames),
                ]
                if not no_mask:
                    cmd.extend(["--mask_img", str(mask_path)])

                stdout_path = video_dir.joinpath("slam_stdout.txt")
                stderr_path = video_dir.joinpath("slam_stderr.txt")

                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    pbar.update(len(completed))

                fut = executor.submit(
                    runner, cmd, str(video_dir), stdout_path, stderr_path, timeout
                )
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
    n_timeout = sum(1 for r in results if isinstance(r, subprocess.TimeoutExpired))
    n_fail = len(results) - n_ok - n_timeout
    logger.info(
        "Done: %d succeeded, %d failed, %d timed out (logs: slam_stdout.txt / slam_stderr.txt)",
        n_ok,
        n_fail,
        n_timeout,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
