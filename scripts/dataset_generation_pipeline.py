"""Run the full dataset generation pipeline for one or more session directories.

:Steps:
    0. Organize raw GoPro MP4 files into the demo directory structure.
    1. Extract IMU telemetry from each raw_video.mp4.
    2. Build an ORB-SLAM3 map atlas from the mapping video.
    3. Run ORB-SLAM3 localization on all demo videos.
    4. Detect ArUco tags in all demo videos.
    5. Run SLAM tag and gripper range calibrations.
    6. Generate dataset_plan.pkl.
    7. Generate the dataset (MCAP or Zarr replay buffer).

:Usage:
    uv run python scripts/dataset_generation_pipeline.py <session_dir> [<session_dir> ...]
"""

import logging
import pathlib
import subprocess
import sys

import click

logger = logging.getLogger(__name__)


def _run(cmd: list, step_name: str) -> None:
    """Run a subprocess command, re-raising failures as a ClickException with step context."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise click.ClickException(
            f"Step '{step_name}' failed with exit code {e.returncode}."
        ) from e


# TODO(abhichothani42): make these scripts import modules
@click.command()
@click.argument("session_dir", nargs=-1, required=True)
@click.option("-c", "--calibration_dir", default=None)
@click.option(
    "-f",
    "--format",
    "dataset_format",
    type=click.Choice(["mcap", "zarr"], case_sensitive=False),
    default="mcap",
    show_default=True,
    help="Output dataset format.",
)
@click.option(
    "-sfs",
    "--slam_frame_stride",
    type=int,
    default=2,
    help="Frame stride used by SLAM and ArUco (raw_fps / slam_fps).",
)
def main(session_dir, calibration_dir, dataset_format, slam_frame_stride):
    """Run the full dataset generation pipeline for each session directory.

    :param session_dir: One or more session directories to process.
    :param calibration_dir: Path to the calibration directory containing
        gopro13_intrinsics_2_7k.json and aruco_config.yaml. Defaults to
        example/calibration/ relative to the repository root.
    :param dataset_format: Output format — 'mcap' (default) or 'zarr'.
    :param slam_frame_stride: Frame stride matching SLAM/ArUco detection rate
        (raw_fps / slam_fps). Default 2 matches 120 fps -> 60 fps SLAM.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    script_dir = pathlib.Path(__file__).parent.joinpath("scripts_slam_pipeline")
    if calibration_dir is None:
        calibration_dir = (
            pathlib.Path(__file__)
            .parent.parent.joinpath("example", "calibration")
            .resolve()
        )
    else:
        calibration_dir = pathlib.Path(calibration_dir).resolve()
    if not calibration_dir.is_dir():
        raise click.ClickException(
            f"Calibration directory not found: {calibration_dir}"
        )

    for session in session_dir:
        session = pathlib.Path(session).resolve()

        if not session.is_dir():
            raise click.ClickException(f"Session directory not found: {session}")

        # 00 Organize raw GoPro videos into demo directory structure
        logger.info("\n%s 00_process_videos %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("00_process_videos.py")

        if not script_path.is_file():
            raise click.ClickException(f"Could not find script at: {script_path}")

        cmd = [sys.executable, str(script_path), str(session)]
        _run(cmd, "00_process_videos")

        # 01 extract IMU telemetry from each raw_video.mp4 into imu_data.json
        logger.info("\n%s 01_extract_gopro_imu %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("01_extract_gopro_imu.py")

        if not script_path.is_file():
            raise click.ClickException(f"Could not find script at: {script_path}")

        cmd = [sys.executable, str(script_path), str(session)]
        _run(cmd, "01_extract_gopro_imu")

        # 02 create ORB-SLAM3 map atlas from the mapping video
        logger.info("\n%s 02_create_map %s", "#" * 15, "#" * 15)
        logger.info("Depending on your hardware this might take a few minutes")
        script_path = script_dir.joinpath("02_create_map.py")

        if not script_path.is_file():
            raise click.ClickException(f"Could not find script at: {script_path}")

        demo_dir = session.joinpath("demos")
        if not demo_dir.is_dir():
            raise click.ClickException(f"Demos directory not found: {demo_dir}")

        mapping_dir = next(demo_dir.glob("mapping_*"), None)

        if mapping_dir is None or not mapping_dir.is_dir():
            raise click.ClickException(f"No mapping_* directory found in {demo_dir}")

        map_path = mapping_dir.joinpath("map_atlas.osa")

        if not map_path.is_file():
            cmd = [
                sys.executable,
                str(script_path),
                "--input_dir",
                str(mapping_dir),
                "--map_path",
                str(map_path),
            ]
            _run(cmd, "02_create_map")

            # check if map_atlas.osa file is generated
            if not map_path.is_file():
                stdout_log = mapping_dir / "slam_stdout_mapping.txt"
                stderr_log = mapping_dir / "slam_stderr_mapping.txt"
                raise click.ClickException(
                    f"ORB-SLAM3 did not produce map_atlas.osa at: {map_path}\n"
                    f"Check logs for details:\n"
                    f"  stdout: {stdout_log}\n"
                    f"  stderr: {stderr_log}"
                )

        # 03 run SLAM localization on all demo videos using the map atlas
        logger.info("\n%s 03_batch_slam %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("03_batch_slam.py")

        if not script_path.is_file():
            raise click.ClickException(f"Could not find script at: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(demo_dir),
            "--map_path",
            str(map_path),
        ]
        _run(cmd, "03_batch_slam")

        # 04 detect and localize ArUco tags in all demo videos
        logger.info("\n%s 04_detect_aruco %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("04_detect_aruco.py")

        if not script_path.is_file():
            raise click.ClickException(f"Could not find script at: {script_path}")

        camera_intrinsics = calibration_dir.joinpath("gopro13_intrinsics_2_7k.json")
        aruco_config = calibration_dir.joinpath("aruco_config.yaml")

        if not camera_intrinsics.is_file():
            raise click.ClickException(
                f"Camera intrinsics not found: {camera_intrinsics}"
            )
        if not aruco_config.is_file():
            raise click.ClickException(f"ArUco config not found: {aruco_config}")

        cmd = [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(demo_dir),
            "--camera_intrinsics",
            str(camera_intrinsics),
            "--aruco_yaml",
            str(aruco_config),
            "--slam_frame_stride",
            str(slam_frame_stride),
        ]
        _run(cmd, "04_detect_aruco")

        # 05 run slam tag and gripper range calibrations
        logger.info("\n%s 05_run_calibrations %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("05_run_calibrations.py")
        if not script_path.is_file():
            raise click.ClickException(f"Could not find script at: {script_path}")
        cmd = [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(session),
        ]
        _run(cmd, "05_run_calibrations")

        # 06 generate dataset plan
        logger.info("\n%s 06_generate_dataset_plan %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        if not script_path.is_file():
            raise click.ClickException(f"Could not find script at: {script_path}")
        cmd = [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(session),
            "--slam_frame_stride",
            str(slam_frame_stride),
        ]
        _run(cmd, "06_generate_dataset_plan")

        # 07 generate dataset
        logger.info(
            "\n%s 07_generate_dataset (%s) %s", "#" * 15, dataset_format, "#" * 15
        )
        if dataset_format == "mcap":
            script_path = script_dir.joinpath("07_generate_mcap_dataset.py")
            if not script_path.is_file():
                raise click.ClickException(f"Could not find script at: {script_path}")
            output_path = session.joinpath("dataset_mcap")
            cmd = [
                sys.executable,
                str(script_path),
                "--output",
                str(output_path),
                "--slam_frame_stride",
                str(slam_frame_stride),
                str(session),
            ]
        else:
            script_path = script_dir.joinpath("07_generate_zarr_dataset.py")
            if not script_path.is_file():
                raise click.ClickException(f"Could not find script at: {script_path}")
            output_path = session.joinpath("dataset.zarr.zip")
            cmd = [
                sys.executable,
                str(script_path),
                "--output",
                str(output_path),
                "--slam_frame_stride",
                str(slam_frame_stride),
                str(session),
            ]
        _run(cmd, "07_generate_dataset")


if __name__ == "__main__":
    main()
