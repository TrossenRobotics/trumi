"""
Extract GoPro IMU telemetry from raw_video.mp4 files and save as imu_data.json.

:Steps:
    1. Scan <session_dir>/demos/ for video dirs containing raw_video.mp4.
    2. Skip dirs that already have imu_data.json.
    3. Extract ACCL, GYRO, GPS and other GPMF streams in parallel using ThreadPoolExecutor.
    4. Save telemetry as imu_data.json alongside each raw_video.mp4.

:Usage:
    uv run python scripts_slam_pipeline/01_extract_gopro_imu.py <session_dir> [<session_dir> ...]
"""

import concurrent.futures
import multiprocessing
import pathlib
import sys

import click
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
from tqdm import tqdm


def gpmf_extraction(video_input_path, json_output_path):
    """Extract GPMF telemetry streams from a GoPro MP4 and write them to a JSON file.

    :param video_input_path: Path to the source raw_video.mp4 file.
    :param json_output_path: Path where the output imu_data.json will be written.
    :raises SystemExit: If the GPMD source cannot be opened.
    """
    extractor = GoProTelemetryExtractor(video_input_path)

    extractor.open_source()
    if not extractor.handle:
        print(f"Error: Failed to open GPMD source in {video_input_path}")
        sys.exit(1)

    try:
        extractor.extract_data_to_json(
            json_output_path,
            [
                "ACCL",
                "GYRO",
                "GPS5",
                "GPSP",
                "GPSU",
                "GPSF",
                "GRAV",
                "MAGN",
                "CORI",
                "IORI",
            ],
        )
    finally:
        extractor.close_source()


@click.command(
    help="Extract IMU data from one or more SESSION_DIR paths. "
    "Expects MP4s under <session_dir>/raw_videos/. "
)
@click.option("-n", "--num_workers", type=int, default=None)
@click.argument("session_dir", nargs=-1, required=True)
def main(num_workers, session_dir):
    """Extract GoPro IMU data for one or more session directories in parallel.

    Scans <session_dir>/demos/*/raw_video.mp4 and runs GPMF extraction on each,
    writing imu_data.json alongside the video.

    :param num_workers: Number of parallel threads. Defaults to CPU count.
    :param session_dir: One or more session directory paths to process.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    for session in session_dir:
        input_dir = pathlib.Path(session).resolve().joinpath("demos")
        input_video_dirs = [x.parent for x in input_dir.glob("*/raw_video.mp4")]
        print(f"Found {len(input_video_dirs)} video dirs")

        with tqdm(total=len(input_video_dirs)) as pbar:
            # one video per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = set()
                for video_dir in input_video_dirs:
                    video_dir = video_dir.resolve()
                    video_path = video_dir.joinpath("raw_video.mp4")
                    json_path = video_dir.joinpath("imu_data.json")

                    if json_path.is_file():
                        print(
                            f"imu_data.json already exists, skipping {video_dir.name}"
                        )
                        pbar.update(1)
                        continue

                    # throttle: wait for a slot before submitting the next task
                    if len(futures) >= num_workers:
                        completed, futures = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        pbar.update(len(completed))

                    futures.add(
                        executor.submit(
                            gpmf_extraction, str(video_path), str(json_path)
                        )
                    )

                # wait for the last batch to finish
                if futures:
                    completed, _ = concurrent.futures.wait(futures)
                    pbar.update(len(completed))

        print("Done, IMU extraction complete.")


if __name__ == "__main__":
    main()
