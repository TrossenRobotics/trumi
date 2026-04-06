"""
Extract GoPro IMU telemetry from raw_video.mp4 files and save as imu_data.json.

:Steps:
    1. Scan <session_dir>/demos/ for video dirs containing raw_video.mp4.
    2. Skip dirs that already have imu_data.json.
    3. Process multiple videos in parallel using ProcessPoolExecutor, extracting
       ACCL, GYRO, GPS and other GPMF streams from each.
    4. Save telemetry as imu_data.json alongside each raw_video.mp4.

:Usage:
    uv run python scripts_slam_pipeline/01_extract_gopro_imu.py <session_dir> [<session_dir> ...]
"""

import concurrent.futures
import logging
import multiprocessing
import pathlib

import click
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
from tqdm import tqdm

logger = logging.getLogger(__name__)

# GPMF telemetry streams to extract
GPMF_STREAMS = [
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
]


class GPMFExtractionError(Exception):
    """Raised when GPMF extraction fails."""

    pass


def gpmf_extraction(video_input_path, json_output_path):
    """Extract GPMF telemetry streams from a GoPro MP4 and write them to a JSON file.

    :param video_input_path: Path to the source raw_video.mp4 file.
    :param json_output_path: Path where the output imu_data.json will be written.
    :raises GPMFExtractionError: If the GPMD source cannot be opened.
    """
    extractor = GoProTelemetryExtractor(video_input_path)

    extractor.open_source()
    if not extractor.handle:
        raise GPMFExtractionError(f"Failed to open GPMD source in {video_input_path}")

    try:
        extractor.extract_data_to_json(json_output_path, GPMF_STREAMS)
    finally:
        extractor.close_source()


@click.command(
    help="Extract IMU data from one or more SESSION_DIR paths. "
    "Expects MP4s under <session_dir>/demos/*/raw_video.mp4. "
)
@click.option(
    "-n",
    "--num_workers",
    type=click.IntRange(min=1),
    default=None,
    help="Number of parallel processes. Defaults to CPU count.",
)
@click.argument("session_dir", nargs=-1, required=True)
def main(num_workers, session_dir):
    """Extract GoPro IMU data for one or more session directories in parallel.

    Scans <session_dir>/demos/*/raw_video.mp4 and runs GPMF extraction on each,
    writing imu_data.json alongside the video.

    :param num_workers: Number of parallel processes. Defaults to CPU count.
    :param session_dir: One or more session directory paths to process.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    failed_extractions = []

    for session in session_dir:
        session = pathlib.Path(session).resolve()

        if not session.is_dir():
            raise click.ClickException(f"Session directory not found: '{session}'")

        input_dir = session.joinpath("demos")
        if not input_dir.is_dir():
            logger.warning(f"No demos/ directory found in '{session}', skipping.")
            continue

        input_video_dirs = [x.parent for x in input_dir.glob("*/raw_video.mp4")]
        if not input_video_dirs:
            logger.warning(f"No raw_video.mp4 files found in '{input_dir}', skipping.")
            continue

        logger.info(f"Found {len(input_video_dirs)} video dirs")

        with tqdm(total=len(input_video_dirs)) as pbar:
            # one video per process for true parallelism (bypasses GIL)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = {}
                for video_dir in input_video_dirs:
                    video_dir = video_dir.resolve()
                    video_path = video_dir.joinpath("raw_video.mp4")
                    json_path = video_dir.joinpath("imu_data.json")

                    if json_path.is_file():
                        logger.debug(
                            f"imu_data.json already exists, skipping {video_dir.name}"
                        )
                        pbar.update(1)
                        continue

                    future = executor.submit(
                        gpmf_extraction, str(video_path), str(json_path)
                    )
                    futures[future] = video_path

                # collect results and handle exceptions
                for future in concurrent.futures.as_completed(futures):
                    video_path = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(f"Extraction failed for {video_path}: {exc}")
                        failed_extractions.append((video_path, exc))
                    pbar.update(1)

    if failed_extractions:
        logger.error(f"{len(failed_extractions)} extraction(s) failed.")
        raise SystemExit(1)

    logger.info("Done, IMU extraction complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
